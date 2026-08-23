// SPDX-License-Identifier: GPL-2.0
/*
 * BRC checkpoint support for ext4.
 *
 * Phase 3B-1:
 *   - persistent per-inode BRC identity
 *   - checkpoint sealing
 *   - immutable protection
 *
 * No extent sharing is implemented in this phase.
 */

#include <linux/fs.h>
#include <linux/iversion.h>
#include <linux/pagemap.h>
#include <linux/xattr.h>

#include "ext4.h"
#include "ext4_jbd2.h"
#include "xattr.h"
#include "brc.h"

#define EXT4_BRC_XATTR_NAME          "brc"
#define EXT4_BRC_META_MAGIC          0x31435242U /* "BRC1" */
#define EXT4_BRC_META_VERSION        1

#define EXT4_BRC_STATE_BUILDING      1
#define EXT4_BRC_STATE_SEALED        2

/*
 * Persistent BRC metadata stored in trusted.brc.
 *
 * lineage and generation are reserved for Phase 3B-2.
 *
 * No layout_id is stored.  The current BRC prototype relies on
 * the fixed-layout checkpoint invariant defined by the design.
 */
struct ext4_brc_meta_disk {
        __le32 magic;
        __le16 version;
        __le16 state;
        __le64 lineage_hi;
        __le64 lineage_lo;
        __le64 generation;
        __le64 reserved;
} __packed;


/*
 * Return:
 *   1  inode has trusted.brc
 *   0  inode is not BRC managed
 *  <0  error
 */
int ext4_brc_has_marker(struct inode *inode)
{
        int ret;

        ret = ext4_xattr_get(inode,
                             EXT4_XATTR_INDEX_TRUSTED,
                             EXT4_BRC_XATTR_NAME,
                             NULL, 0);

        if (ret == -ENODATA)
                return 0;

        if (ret < 0)
                return ret;

        return 1;
}


static int ext4_brc_set_immutable(struct inode *inode)
{
        handle_t *handle;
        struct ext4_iloc iloc;
        int ret;
        int stop_ret;

        if (IS_IMMUTABLE(inode))
                return 0;

        /*
         * Follow the ordering already used by ext4_ioctl_setflags():
         * wait for direct I/O and flush dirty pages before publishing
         * the immutable state.
         */
        inode_dio_wait(inode);

        ret = filemap_write_and_wait(inode->i_mapping);
        if (ret)
                return ret;

        handle = ext4_journal_start(inode, EXT4_HT_INODE, 1);
        if (IS_ERR(handle))
                return PTR_ERR(handle);

        if (IS_SYNC(inode))
                ext4_handle_sync(handle);

        ret = ext4_reserve_inode_write(handle, inode, &iloc);
        if (ret)
                goto out_stop;

        ext4_set_inode_flag(inode, EXT4_INODE_IMMUTABLE);
        ext4_set_inode_flags(inode, false);

        inode_set_ctime_current(inode);
        inode_inc_iversion(inode);

        ret = ext4_mark_iloc_dirty(handle, inode, &iloc);

out_stop:
        stop_ret = ext4_journal_stop(handle);
        if (!ret)
                ret = stop_ret;

        return ret;
}


int ext4_brc_seal_inode(struct inode *inode)
{
        struct ext4_brc_meta_disk meta = { 0 };
        int ret;

        if (!S_ISREG(inode->i_mode))
                return -EINVAL;

        /*
         * Do not silently convert a normal file that was already made
         * immutable by userspace into a BRC checkpoint.
         */
        if (IS_IMMUTABLE(inode))
                return -EPERM;

        ret = ext4_brc_has_marker(inode);
        if (ret < 0)
                return ret;

        if (ret > 0)
                return -EEXIST;

        meta.magic = cpu_to_le32(EXT4_BRC_META_MAGIC);
        meta.version = cpu_to_le16(EXT4_BRC_META_VERSION);
        meta.state = cpu_to_le16(EXT4_BRC_STATE_SEALED);

        /*
         * trusted.brc is the persistent per-inode BRC identity.
         * lineage fields remain zero until Phase 3B-2.
         */
        ret = ext4_xattr_set(inode,
                             EXT4_XATTR_INDEX_TRUSTED,
                             EXT4_BRC_XATTR_NAME,
                             &meta, sizeof(meta),
                             XATTR_CREATE);
        if (ret)
                return ret;

        ret = ext4_brc_set_immutable(inode);

        /*
         * Phase 3B-1 contains no shared physical blocks.
         * If immutable publication fails, remove the marker instead of
         * leaving an ordinary writable inode falsely registered as BRC.
         *
         * Crash atomicity between these two metadata transitions will
         * be addressed before physical sharing is enabled.
         */
        if (ret && !IS_IMMUTABLE(inode))
                ext4_xattr_set(inode,
                               EXT4_XATTR_INDEX_TRUSTED,
                               EXT4_BRC_XATTR_NAME,
                               NULL, 0,
                               XATTR_REPLACE);

        return ret;
}
