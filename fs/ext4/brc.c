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
#include <linux/anon_inodes.h>
#include <linux/file.h>
#include <linux/mutex.h>
#include <linux/random.h>
#include <linux/slab.h>
#include <linux/iversion.h>
#include <linux/pagemap.h>
#include <linux/xattr.h>

#include "ext4.h"
#include "ext4_jbd2.h"
#include "xattr.h"
#include "brc.h"


/*
 * A live BRC session represents the capability to extend one
 * checkpoint inheritance lineage.
 *
 * The session contains only constant-size live state.  It does not
 * maintain an in-memory list of all checkpoints in the lineage.
 */
struct ext4_brc_session {
        /*
         * Pin the originating ext4 file and mount for the lifetime
         * of the anonymous session fd.
         */
        struct file *anchor_file;

        /* Random 128-bit identity of this live lineage. */
        u8 lineage[16];

        /*
         * Current generation.  It becomes meaningful once tail_inode
         * is established by the first legal root seal.
         */
        u64 generation;

        /*
         * At most one sealed tail and one building child are pinned.
         * Session memory therefore remains O(1).
         */
        struct inode *tail_inode;
        struct inode *building_inode;

        struct mutex lock;
};


static int ext4_brc_session_release(struct inode *inode,
                                    struct file *file)
{
        struct ext4_brc_session *session = file->private_data;
        struct super_block *sb;

        if (!session)
                return 0;

        sb = file_inode(session->anchor_file)->i_sb;

        ext4_msg(sb, KERN_INFO, "BRC_SESSION_RELEASE");

        if (session->building_inode)
                iput(session->building_inode);

        if (session->tail_inode)
                iput(session->tail_inode);

        fput(session->anchor_file);
        kfree(session);

        return 0;
}


static const struct file_operations ext4_brc_session_fops = {
        .release = ext4_brc_session_release,
};


static struct ext4_brc_session *
ext4_brc_session_from_fd(int session_fd, struct fd *session_f)
{
        struct file *file;

        *session_f = fdget(session_fd);
        file = fd_file(*session_f);

        if (!file)
                return ERR_PTR(-EBADF);

        if (file->f_op != &ext4_brc_session_fops) {
                fdput(*session_f);
                return ERR_PTR(-EINVAL);
        }

        if (!file->private_data) {
                fdput(*session_f);
                return ERR_PTR(-EINVAL);
        }

        return file->private_data;
}


int ext4_brc_session_begin(struct file *anchor_file)
{
        struct inode *inode = file_inode(anchor_file);
        struct ext4_brc_session *session;
        int fd;

        if (!S_ISDIR(inode->i_mode))
                return -ENOTDIR;

        session = kzalloc(sizeof(*session), GFP_KERNEL);
        if (!session)
                return -ENOMEM;

        /*
         * Keep the ext4 mount/filesystem referenced for as long as
         * the anonymous BRC session fd exists.
         */
        get_file(anchor_file);
        session->anchor_file = anchor_file;

        get_random_bytes(session->lineage,
                         sizeof(session->lineage));

        mutex_init(&session->lock);

        /*
         * A newly-created session has no checkpoint yet.
         * The first legal BRC_SEAL will later establish generation 0.
         */
        session->generation = 0;
        session->tail_inode = NULL;
        session->building_inode = NULL;

        fd = anon_inode_getfd("[ext4-brc-session]",
                              &ext4_brc_session_fops,
                              session,
                              O_RDWR | O_CLOEXEC);
        if (fd < 0) {
                fput(session->anchor_file);
                kfree(session);
                return fd;
        }

        ext4_msg(inode->i_sb, KERN_INFO,
                 "BRC_SESSION_BEGIN: anchor_inode=%lu session_fd=%d",
                 inode->i_ino, fd);

        return fd;
}


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
static int ext4_brc_read_meta(struct inode *inode,
                              struct ext4_brc_meta_disk *meta)
{
        int ret;

        ret = ext4_xattr_get(inode,
                             EXT4_XATTR_INDEX_TRUSTED,
                             EXT4_BRC_XATTR_NAME,
                             meta, sizeof(*meta));

        if (ret == -ENODATA)
                return 0;

        if (ret < 0)
                return ret;

        if (ret != sizeof(*meta))
                return -EINVAL;

        return 1;
}


static int ext4_brc_write_meta(struct inode *inode,
                               struct ext4_brc_meta_disk *meta,
                               int flags)
{
        return ext4_xattr_set(inode,
                              EXT4_XATTR_INDEX_TRUSTED,
                              EXT4_BRC_XATTR_NAME,
                              meta, sizeof(*meta),
                              flags);
}


int ext4_brc_has_marker(struct inode *inode)
{
        struct ext4_brc_meta_disk meta;

        return ext4_brc_read_meta(inode, &meta);
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


static void ext4_brc_meta_init(struct ext4_brc_meta_disk *meta,
                               struct ext4_brc_session *session,
                               u16 state,
                               u64 generation)
{
        memset(meta, 0, sizeof(*meta));

        meta->magic = cpu_to_le32(EXT4_BRC_META_MAGIC);
        meta->version = cpu_to_le16(EXT4_BRC_META_VERSION);
        meta->state = cpu_to_le16(state);

        /*
         * lineage_hi and lineage_lo are treated as one opaque
         * 128-bit token.  No numeric interpretation is required.
         */
        memcpy(&meta->lineage_hi,
               session->lineage,
               sizeof(meta->lineage_hi));
        memcpy(&meta->lineage_lo,
               session->lineage + sizeof(meta->lineage_hi),
               sizeof(meta->lineage_lo));

        meta->generation = cpu_to_le64(generation);
}


static bool ext4_brc_same_lineage(struct ext4_brc_meta_disk *meta,
                                  struct ext4_brc_session *session)
{
        return !memcmp(&meta->lineage_hi,
                       session->lineage,
                       sizeof(meta->lineage_hi)) &&
               !memcmp(&meta->lineage_lo,
                       session->lineage + sizeof(meta->lineage_hi),
                       sizeof(meta->lineage_lo));
}


int ext4_brc_prepare_child(struct file *child_file,
                           struct file *parent_file,
                           int session_fd)
{
        struct inode *child_inode = file_inode(child_file);
        struct inode *parent_inode = file_inode(parent_file);
        struct ext4_brc_session *session;
        struct ext4_brc_meta_disk parent_meta;
        struct ext4_brc_meta_disk child_meta;
        struct fd session_f = { };
        struct inode *building_ref = NULL;
        u64 child_generation;
        int ret;

        session = ext4_brc_session_from_fd(session_fd, &session_f);
        if (IS_ERR(session))
                return PTR_ERR(session);

        mutex_lock(&session->lock);

        if (file_inode(session->anchor_file)->i_sb != child_inode->i_sb) {
                ret = -EXDEV;
                goto out_unlock;
        }

        if (parent_inode->i_sb != child_inode->i_sb) {
                ret = -EXDEV;
                goto out_unlock;
        }

        if (!session->tail_inode) {
                ret = -EINVAL;
                goto out_unlock;
        }

        if (session->building_inode) {
                ret = -EBUSY;
                goto out_unlock;
        }

        if (session->tail_inode != parent_inode) {
                ret = -EPERM;
                goto out_unlock;
        }

        ret = ext4_brc_read_meta(parent_inode, &parent_meta);
        if (ret <= 0) {
                if (!ret)
                        ret = -EINVAL;
                goto out_unlock;
        }

        if (le16_to_cpu(parent_meta.state) != EXT4_BRC_STATE_SEALED) {
                ret = -EINVAL;
                goto out_unlock;
        }

        if (!ext4_brc_same_lineage(&parent_meta, session)) {
                ret = -EPERM;
                goto out_unlock;
        }

        ret = ext4_brc_has_marker(child_inode);
        if (ret < 0)
                goto out_unlock;

        if (ret > 0) {
                ret = -EEXIST;
                goto out_unlock;
        }

        child_generation = session->generation + 1;

        ext4_brc_meta_init(&child_meta,
                           session,
                           EXT4_BRC_STATE_BUILDING,
                           child_generation);

        /*
         * Phase 3B-2 records only the parent-child eligibility state.
         * No physical block remapping occurs here.
         */
        ret = ext4_brc_write_meta(child_inode,
                                  &child_meta,
                                  XATTR_CREATE);
        if (ret)
                goto out_unlock;

        building_ref = igrab(child_inode);
        if (!building_ref) {
                ext4_xattr_set(child_inode,
                               EXT4_XATTR_INDEX_TRUSTED,
                               EXT4_BRC_XATTR_NAME,
                               NULL, 0,
                               XATTR_REPLACE);
                ret = -ESTALE;
                goto out_unlock;
        }

        session->building_inode = building_ref;

        ext4_msg(child_inode->i_sb, KERN_INFO,
                 "BRC_BUILDING: parent_inode=%lu child_inode=%lu generation=%llu",
                 parent_inode->i_ino,
                 child_inode->i_ino,
                 (unsigned long long)child_generation);

        ret = 0;

out_unlock:
        mutex_unlock(&session->lock);
        fdput(session_f);
        return ret;
}


int ext4_brc_seal_with_session(struct file *file,
                               int session_fd)
{
        struct inode *inode = file_inode(file);
        struct ext4_brc_session *session;
        struct ext4_brc_meta_disk meta;
        struct fd session_f = { };
        struct inode *new_tail = NULL;
        struct inode *old_tail = NULL;
        int ret;

        session = ext4_brc_session_from_fd(session_fd, &session_f);
        if (IS_ERR(session))
                return PTR_ERR(session);

        mutex_lock(&session->lock);
        inode_lock(inode);

        if (file_inode(session->anchor_file)->i_sb != inode->i_sb) {
                ret = -EXDEV;
                goto out_inode;
        }

        /*
         * Root case:
         * a fresh session has neither a tail nor a building child.
         * The first explicitly sealed full checkpoint becomes
         * generation 0.
         */
        if (!session->tail_inode && !session->building_inode) {
                ret = ext4_brc_has_marker(inode);
                if (ret < 0)
                        goto out_inode;

                if (ret > 0) {
                        ret = -EEXIST;
                        goto out_inode;
                }

                if (IS_IMMUTABLE(inode)) {
                        ret = -EPERM;
                        goto out_inode;
                }

                new_tail = igrab(inode);
                if (!new_tail) {
                        ret = -ESTALE;
                        goto out_inode;
                }

                ext4_brc_meta_init(&meta,
                                   session,
                                   EXT4_BRC_STATE_SEALED,
                                   0);

                ret = ext4_brc_write_meta(inode,
                                          &meta,
                                          XATTR_CREATE);
                if (ret)
                        goto out_put_new;

                ret = ext4_brc_set_immutable(inode);
                if (ret) {
                        if (!IS_IMMUTABLE(inode))
                                ext4_xattr_set(
                                    inode,
                                    EXT4_XATTR_INDEX_TRUSTED,
                                    EXT4_BRC_XATTR_NAME,
                                    NULL, 0,
                                    XATTR_REPLACE);
                        goto out_put_new;
                }

                session->tail_inode = new_tail;
                new_tail = NULL;
                session->generation = 0;

                ext4_msg(inode->i_sb, KERN_INFO,
                         "BRC_ROOT_SEAL: inode=%lu generation=0",
                         inode->i_ino);

                ret = 0;
                goto out_inode;
        }

        /*
         * Once a session already has a tail, only the one inode
         * previously established by BRC_CREATE may be sealed.
         */
        if (!session->building_inode ||
            session->building_inode != inode) {
                ret = -EPERM;
                goto out_inode;
        }

        ret = ext4_brc_read_meta(inode, &meta);
        if (ret <= 0) {
                if (!ret)
                        ret = -EINVAL;
                goto out_inode;
        }

        if (le16_to_cpu(meta.state) != EXT4_BRC_STATE_BUILDING) {
                ret = -EINVAL;
                goto out_inode;
        }

        if (!ext4_brc_same_lineage(&meta, session)) {
                ret = -EPERM;
                goto out_inode;
        }

        if (le64_to_cpu(meta.generation) !=
            session->generation + 1) {
                ret = -EINVAL;
                goto out_inode;
        }

        /*
         * Phase 3B-3 will insert clean parent mappings here,
         * before the BUILDING -> SEALED transition.
         *
         * Phase 3B-2 deliberately performs no remapping.
         */
        meta.state = cpu_to_le16(EXT4_BRC_STATE_SEALED);

        ret = ext4_brc_write_meta(inode,
                                  &meta,
                                  XATTR_REPLACE);
        if (ret)
                goto out_inode;

        ret = ext4_brc_set_immutable(inode);
        if (ret) {
                meta.state = cpu_to_le16(EXT4_BRC_STATE_BUILDING);
                ext4_brc_write_meta(inode,
                                    &meta,
                                    XATTR_REPLACE);
                goto out_inode;
        }

        old_tail = session->tail_inode;

        /*
         * Transfer the reference already owned by building_inode
         * into tail_inode.  No additional igrab() is needed.
         */
        session->tail_inode = session->building_inode;
        session->building_inode = NULL;
        session->generation++;

        ext4_msg(inode->i_sb, KERN_INFO,
                 "BRC_CHILD_SEAL: inode=%lu generation=%llu",
                 inode->i_ino,
                 (unsigned long long)session->generation);

        ret = 0;

out_inode:
        inode_unlock(inode);

        if (old_tail)
                iput(old_tail);

        if (new_tail)
                iput(new_tail);

        mutex_unlock(&session->lock);
        fdput(session_f);
        return ret;

out_put_new:
        iput(new_tail);
        new_tail = NULL;
        goto out_inode;
}
