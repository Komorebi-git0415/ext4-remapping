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
#include "ext4_extents.h"
#include "extents_status.h"
#include "xattr.h"
#include "brc.h"


/*
 * Persistent per-lineage metadata.
 *
 * Phase 4 separates checkpoint-local identity (trusted.brc) from
 * persistent lineage ordering.  The lineage ledger is an ordinary
 * BRC-managed ext4 regular file.
 */
#define EXT4_BRC_LINEAGE_MAGIC          0x314c5242U /* "BRL1" */
#define EXT4_BRC_LINEAGE_VERSION        1

/*
 * Version-1 lineage header feature flags.
 *
 * Old Phase-4A/4B ledgers have flags == 0 and reserved1 == 0.
 * Such a ledger is interpreted as having an implicit physical
 * reclamation cursor:
 *
 *     reclaim_generation == base_generation
 *
 * Once physical reclamation advances beyond BASE, the cursor feature
 * is made explicit and the final 64-bit header field stores R.
 *
 * An older kernel does not recognize this flag and therefore fails
 * closed instead of operating on a physically modified lineage.
 */
#define EXT4_BRC_LINEAGE_F_RECLAIM_CURSOR       0x0001
#define EXT4_BRC_LINEAGE_KNOWN_FLAGS            \
        EXT4_BRC_LINEAGE_F_RECLAIM_CURSOR

/*
 * Fixed 64-byte on-disk lineage header.
 *
 * tail_generation is derived once entries exist:
 *
 *   base_generation + nr_entries - 1
 */
struct ext4_brc_lineage_header_disk {
        __le32 magic;
        __le16 version;
        __le16 header_size;

        __le16 entry_size;
        __le16 flags;
        __le32 reserved0;

        __le64 lineage_hi;
        __le64 lineage_lo;

        __le64 base_generation;
        __le64 head_generation;
        __le64 nr_entries;

        /*
         * flags == 0:
         *     must be zero; R is implicitly base_generation.
         *
         * EXT4_BRC_LINEAGE_F_RECLAIM_CURSOR:
         *     stores persistent reclaim_generation R.
         */
        __le64 reclaim_generation;
} __packed;


/*
 * Fixed 16-byte checkpoint reference.
 *
 * checkpoint_generation is not duplicated here.  For an uncompacted
 * ledger it is derived from base_generation plus the entry index and
 * is independently verified against trusted.brc when the inode is
 * resolved.
 */
struct ext4_brc_lineage_entry_disk {
        __le64 inode_number;
        __le32 inode_generation;
        __le32 flags;
} __packed;


/*
 * Decode the oldest checkpoint whose physical reclamation is not yet
 * complete.
 *
 * Legacy Phase-4A/4B ledgers did not persist R.  Their zero flags and
 * zero final header field are interpreted as:
 *
 *     R = BASE
 *
 * This preserves the existing 64-byte version-1 on-disk layout while
 * allowing Phase 4C to make R persistent once it actually advances.
 */
static u64 ext4_brc_lineage_reclaim_generation(
        const struct ext4_brc_lineage_header_disk *header)
{
        u16 flags = le16_to_cpu(header->flags);

        if (flags & EXT4_BRC_LINEAGE_F_RECLAIM_CURSOR)
                return le64_to_cpu(
                        header->reclaim_generation);

        return le64_to_cpu(
                header->base_generation);
}


/*
 * Phase-4 prototype serialization for persistent lineage metadata.
 *
 * A standalone reclamation ioctl can run without an ephemeral session,
 * so it cannot use session->lock.  Serialize BRC-controlled ledger
 * append and HEAD advancement here.  This is intentionally coarse
 * grained for the research prototype and can later become per-ledger.
 */
static DEFINE_MUTEX(ext4_brc_lineage_io_mutex);


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

        /*
         * Persistent lineage ledger associated with this session.
         *
         * This is NULL for the legacy Phase-3 SESSION_BEGIN path.
         * For LINEAGE_BEGIN it aliases anchor_file; anchor_file owns
         * the file reference.
         */
        struct file *lineage_file;

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


int ext4_brc_lineage_begin(struct file *lineage_file)
{
        struct inode *inode = file_inode(lineage_file);
        struct ext4_brc_lineage_header_disk header;
        struct ext4_brc_session *session;
        loff_t pos = 0;
        ssize_t written;
        int fd;
        int ret;

        if (!S_ISREG(inode->i_mode))
                return -EINVAL;

        /*
         * The Phase-4A prototype uses an ordinary empty ext4 file as
         * the persistent lineage ledger.
         */
        if (!(lineage_file->f_mode & FMODE_READ) ||
            !(lineage_file->f_mode & FMODE_WRITE))
                return -EBADF;

        if (i_size_read(inode) != 0)
                return -EINVAL;

        session = kzalloc(sizeof(*session), GFP_KERNEL);
        if (!session)
                return -ENOMEM;

        /*
         * Pin the ledger file, and therefore the ext4 mount, for the
         * lifetime of the anonymous session fd.
         */
        get_file(lineage_file);
        session->anchor_file = lineage_file;
        session->lineage_file = lineage_file;

        get_random_bytes(session->lineage,
                         sizeof(session->lineage));

        mutex_init(&session->lock);

        session->generation = 0;
        session->tail_inode = NULL;
        session->building_inode = NULL;

        memset(&header, 0, sizeof(header));

        header.magic =
                cpu_to_le32(EXT4_BRC_LINEAGE_MAGIC);
        header.version =
                cpu_to_le16(EXT4_BRC_LINEAGE_VERSION);
        header.header_size =
                cpu_to_le16((u16)sizeof(header));
        header.entry_size =
                cpu_to_le16(
                        (u16)sizeof(struct ext4_brc_lineage_entry_disk));

        /*
         * lineage_hi/lo are an opaque persistent copy of the same
         * 128-bit identity carried by trusted.brc checkpoints.
         */
        memcpy(&header.lineage_hi,
               session->lineage,
               sizeof(header.lineage_hi));
        memcpy(&header.lineage_lo,
               session->lineage + sizeof(header.lineage_hi),
               sizeof(header.lineage_lo));

        /*
         * The ledger initially contains no checkpoint entries.
         * Generation zero becomes real when the root checkpoint is
         * successfully sealed and appended in the next Phase-4A step.
         */
        header.base_generation = cpu_to_le64(0);
        header.head_generation = cpu_to_le64(0);
        header.nr_entries = cpu_to_le64(0);

        written = kernel_write(lineage_file,
                               &header,
                               sizeof(header),
                               &pos);
        if (written < 0) {
                ret = (int)written;
                goto out_session;
        }

        if (written != (ssize_t)sizeof(header)) {
                ret = -EIO;
                goto out_session;
        }

        /*
         * Make the initial lineage object durable before publishing
         * the anonymous session capability.
         */
        ret = vfs_fsync(lineage_file, 0);
        if (ret)
                goto out_session;

        fd = anon_inode_getfd("[ext4-brc-lineage-session]",
                              &ext4_brc_session_fops,
                              session,
                              O_RDWR | O_CLOEXEC);
        if (fd < 0) {
                ret = fd;
                goto out_session;
        }

        ext4_msg(inode->i_sb, KERN_INFO,
                 "BRC_LINEAGE_BEGIN: ledger_inode=%lu session_fd=%d",
                 inode->i_ino, fd);

        return fd;

out_session:
        fput(session->anchor_file);
        kfree(session);
        return ret;
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


/*
 * Read exactly @len bytes from a BRC-managed metadata file.
 */
static int ext4_brc_file_read_exact(struct file *file,
                                    void *buf,
                                    size_t len,
                                    loff_t pos)
{
        u8 *p = buf;
        size_t done = 0;

        while (done < len) {
                ssize_t ret;

                ret = kernel_read(file,
                                  p + done,
                                  len - done,
                                  &pos);
                if (ret < 0)
                        return ret;

                if (!ret)
                        return -EIO;

                done += ret;
        }

        return 0;
}


/*
 * Write exactly @len bytes to a BRC-managed metadata file.
 */
static int ext4_brc_file_write_exact(struct file *file,
                                     const void *buf,
                                     size_t len,
                                     loff_t pos)
{
        const u8 *p = buf;
        size_t done = 0;

        while (done < len) {
                ssize_t ret;

                ret = kernel_write(file,
                                   p + done,
                                   len - done,
                                   &pos);
                if (ret < 0)
                        return ret;

                if (!ret)
                        return -EIO;

                done += ret;
        }

        return 0;
}


static int
ext4_brc_lineage_validate_header_format(
        const struct ext4_brc_lineage_header_disk *header)
{
        u16 flags;
        u64 base;
        u64 reclaim_generation;
        u64 head;
        u64 nr_entries;
        u64 tail;

        if (le32_to_cpu(header->magic) !=
            EXT4_BRC_LINEAGE_MAGIC)
                return -EFSCORRUPTED;

        if (le16_to_cpu(header->version) !=
            EXT4_BRC_LINEAGE_VERSION)
                return -EFSCORRUPTED;

        if (le16_to_cpu(header->header_size) !=
            sizeof(*header))
                return -EFSCORRUPTED;

        if (le16_to_cpu(header->entry_size) !=
            sizeof(struct ext4_brc_lineage_entry_disk))
                return -EFSCORRUPTED;

        flags = le16_to_cpu(header->flags);

        if (flags & ~EXT4_BRC_LINEAGE_KNOWN_FLAGS)
                return -EFSCORRUPTED;

        if (le32_to_cpu(header->reserved0))
                return -EFSCORRUPTED;

        /*
         * Legacy version-1 ledgers have no explicit reclaim cursor.
         * Their final header field must remain zero and R is inferred
         * to equal BASE.
         */
        if (!(flags &
              EXT4_BRC_LINEAGE_F_RECLAIM_CURSOR) &&
            le64_to_cpu(header->reclaim_generation))
                return -EFSCORRUPTED;

        base = le64_to_cpu(header->base_generation);
        reclaim_generation =
                ext4_brc_lineage_reclaim_generation(header);
        head = le64_to_cpu(header->head_generation);
        nr_entries = le64_to_cpu(header->nr_entries);

        if (!nr_entries) {
                /*
                 * Empty lineage:
                 *
                 *     B = R = H
                 */
                if (head != base ||
                    reclaim_generation != base)
                        return -EFSCORRUPTED;

                return 0;
        }

        /*
         * tail_generation is derived rather than persisted:
         *
         *     tail = base + nr_entries - 1
         */
        tail = base + nr_entries - 1;
        if (tail < base)
                return -EFSCORRUPTED;

        /*
         * Phase-4C persistent lineage invariant:
         *
         *     B <= R <= H <= T
         */
        if (head < base || head > tail)
                return -EFSCORRUPTED;

        if (reclaim_generation < base ||
            reclaim_generation > head)
                return -EFSCORRUPTED;

        return 0;
}


static int
ext4_brc_lineage_validate_session_header(
        struct ext4_brc_session *session,
        const struct ext4_brc_lineage_header_disk *header)
{
        int ret;

        ret = ext4_brc_lineage_validate_header_format(header);
        if (ret)
                return ret;

        /*
         * Session-driven append additionally requires the ledger to
         * carry the same opaque 128-bit lineage identity.
         */
        if (memcmp(&header->lineage_hi,
                   session->lineage,
                   sizeof(header->lineage_hi)) ||
            memcmp(&header->lineage_lo,
                   session->lineage +
                   sizeof(header->lineage_hi),
                   sizeof(header->lineage_lo)))
                return -EPERM;

        return 0;
}

/*
 * Publish one sealed checkpoint into the persistent lineage ledger.
 *
 * The operation is idempotent for the current tail entry.  This is
 * important because the checkpoint is deliberately marked SEALED
 * before its lineage membership is published.  If publication returns
 * an error, a later BRC_SEAL retry may safely attempt the same append.
 *
 * Publication order inside the ledger is:
 *
 *     entry contents -> fsync -> nr_entries -> fsync
 *
 * Thus nr_entries never intentionally exposes an entry whose contents
 * have not first been made durable.
 */
static int
__ext4_brc_lineage_append_checkpoint(
        struct ext4_brc_session *session,
        struct inode *inode,
        u64 checkpoint_generation)
{
        struct ext4_brc_lineage_header_disk header;
        struct ext4_brc_lineage_entry_disk entry;
        struct ext4_brc_lineage_entry_disk existing;
        struct file *file = session->lineage_file;
        u64 base;
        u64 nr_entries;
        u64 index;
        __le64 new_nr_entries;
        loff_t entry_pos;
        loff_t count_pos;
        int ret;

        /*
         * Preserve the verified Phase-3 interface.  Legacy sessions
         * have no persistent lineage file and therefore perform no
         * Phase-4A ledger update.
         */
        if (!file)
                return 0;

        ret = ext4_brc_file_read_exact(file,
                                       &header,
                                       sizeof(header),
                                       0);
        if (ret)
                return ret;

        ret = ext4_brc_lineage_validate_session_header(session,
                                               &header);
        if (ret)
                return ret;

        base = le64_to_cpu(header.base_generation);
        nr_entries = le64_to_cpu(header.nr_entries);

        if (checkpoint_generation < base)
                return -EINVAL;

        index = checkpoint_generation - base;

        /*
         * Linear inheritance permits no holes in the ordered ledger.
         */
        if (index > nr_entries)
                return -EINVAL;

        if (index < nr_entries) {
                /*
                 * Only the most recently published entry can be a
                 * legitimate retry of the current seal operation.
                 */
                if (index + 1 != nr_entries)
                        return -EEXIST;

                entry_pos =
                        sizeof(header) +
                        index * sizeof(existing);

                ret = ext4_brc_file_read_exact(file,
                                               &existing,
                                               sizeof(existing),
                                               entry_pos);
                if (ret)
                        return ret;

                if (le64_to_cpu(existing.inode_number) !=
                    (u64)inode->i_ino ||
                    le32_to_cpu(existing.inode_generation) !=
                    inode->i_generation ||
                    le32_to_cpu(existing.flags) != 0)
                        return -EEXIST;

                /*
                 * The prior attempt may have returned from fsync with
                 * an error even though the data ultimately reached
                 * stable storage.  Re-fsync before declaring the retry
                 * successful.
                 */
                return vfs_fsync(file, 0);
        }

        memset(&entry, 0, sizeof(entry));

        entry.inode_number =
                cpu_to_le64((u64)inode->i_ino);
        entry.inode_generation =
                cpu_to_le32(inode->i_generation);

        entry_pos =
                sizeof(header) +
                index * sizeof(entry);

        ret = ext4_brc_file_write_exact(file,
                                        &entry,
                                        sizeof(entry),
                                        entry_pos);
        if (ret)
                return ret;

        /*
         * First make the entry durable.  Only then advance the
         * published entry count in the header.
         */
        ret = vfs_fsync(file, 0);
        if (ret)
                return ret;

        new_nr_entries = cpu_to_le64(nr_entries + 1);

        count_pos = offsetof(
                struct ext4_brc_lineage_header_disk,
                nr_entries);

        ret = ext4_brc_file_write_exact(file,
                                        &new_nr_entries,
                                        sizeof(new_nr_entries),
                                        count_pos);
        if (ret)
                return ret;

        ret = vfs_fsync(file, 0);
        if (ret)
                return ret;

        ext4_msg(inode->i_sb, KERN_INFO,
                 "BRC_LINEAGE_APPEND: ledger_inode=%lu checkpoint_inode=%lu generation=%llu nr_entries=%llu",
                 file_inode(file)->i_ino,
                 inode->i_ino,
                 (unsigned long long)checkpoint_generation,
                 (unsigned long long)(nr_entries + 1));

        return 0;
}


static int
ext4_brc_lineage_append_checkpoint(
        struct ext4_brc_session *session,
        struct inode *inode,
        u64 checkpoint_generation)
{
        int ret;

        /*
         * Preserve the Phase-3 legacy session path.
         */
        if (!session->lineage_file)
                return 0;

        mutex_lock(&ext4_brc_lineage_io_mutex);

        ret = __ext4_brc_lineage_append_checkpoint(
                        session,
                        inode,
                        checkpoint_generation);

        mutex_unlock(&ext4_brc_lineage_io_mutex);

        return ret;
}


/*
 * Advance the live-lineage frontier through @through_generation.
 *
 * Phase 4B-1 changes only head_generation:
 *
 *     BASE        unchanged
 *     HEAD        through_generation + 1
 *     nr_entries  unchanged
 *
 * The physical ledger entries are intentionally retained.  Their
 * generation remains:
 *
 *     generation(entry[i]) = base_generation + i
 *
 * Actual checkpoint/block reclamation is added only after this
 * persistent frontier transition has been independently verified.
 */
int ext4_brc_lineage_reclaim_through(
        struct file *lineage_file,
        u64 through_generation)
{
        struct inode *inode = file_inode(lineage_file);
        struct ext4_brc_lineage_header_disk header;
        __le64 new_head_disk;
        u64 base;
        u64 head;
        u64 nr_entries;
        u64 tail;
        u64 new_head;
        loff_t head_pos;
        int ret;

        if (!S_ISREG(inode->i_mode))
                return -EINVAL;

        if (!(lineage_file->f_mode & FMODE_READ) ||
            !(lineage_file->f_mode & FMODE_WRITE))
                return -EBADF;

        /*
         * Positional metadata updates must not be transformed into
         * append writes by an O_APPEND file description.
         */
        if (lineage_file->f_flags & O_APPEND)
                return -EINVAL;

        mutex_lock(&ext4_brc_lineage_io_mutex);

        ret = ext4_brc_file_read_exact(
                        lineage_file,
                        &header,
                        sizeof(header),
                        0);
        if (ret)
                goto out_unlock;

        ret = ext4_brc_lineage_validate_header_format(
                        &header);
        if (ret)
                goto out_unlock;

        base = le64_to_cpu(header.base_generation);
        head = le64_to_cpu(header.head_generation);
        nr_entries = le64_to_cpu(header.nr_entries);

        /*
         * An empty ledger has no live checkpoint that can be retained
         * as the post-reclamation HEAD.
         */
        if (!nr_entries) {
                ret = -ENODATA;
                goto out_unlock;
        }

        tail = base + nr_entries - 1;

        /*
         * Monotonic idempotence.  The requested prefix is already
         * outside the live lineage.
         *
         * Re-fsync before reporting success: an earlier attempt may
         * have updated the page cache and then returned an fsync error.
         */
        if (through_generation < head) {
                ret = vfs_fsync(lineage_file, 0);
                goto out_unlock;
        }

        /*
         * Phase 4B RECLAIM_THROUGH always retains at least one live
         * checkpoint.  Destroying the final tail is a separate future
         * lineage-destruction operation.
         */
        if (through_generation >= tail) {
                ret = -ERANGE;
                goto out_unlock;
        }

        new_head = through_generation + 1;

        /*
         * BASE and nr_entries deliberately remain unchanged.
         */
        new_head_disk = cpu_to_le64(new_head);

        head_pos = offsetof(
                struct ext4_brc_lineage_header_disk,
                head_generation);

        ret = ext4_brc_file_write_exact(
                        lineage_file,
                        &new_head_disk,
                        sizeof(new_head_disk),
                        head_pos);
        if (ret)
                goto out_unlock;

        ret = vfs_fsync(lineage_file, 0);
        if (ret)
                goto out_unlock;

        ext4_msg(inode->i_sb, KERN_INFO,
                 "BRC_RECLAIM_HEAD: ledger_inode=%lu old_head=%llu new_head=%llu tail=%llu base=%llu nr_entries=%llu",
                 inode->i_ino,
                 (unsigned long long)head,
                 (unsigned long long)new_head,
                 (unsigned long long)tail,
                 (unsigned long long)base,
                 (unsigned long long)nr_entries);

        ret = 0;

out_unlock:
        mutex_unlock(&ext4_brc_lineage_io_mutex);
        return ret;
}



/*
 * Read one fixed-size persistent checkpoint reference.
 *
 * entry[i] represents:
 *
 *     generation = base_generation + i
 */
static int ext4_brc_lineage_read_entry(
        struct file *lineage_file,
        u64 index,
        struct ext4_brc_lineage_entry_disk *entry)
{
        loff_t pos;

        pos = sizeof(struct ext4_brc_lineage_header_disk) +
              index *
              sizeof(struct ext4_brc_lineage_entry_disk);

        return ext4_brc_file_read_exact(
                        lineage_file,
                        entry,
                        sizeof(*entry),
                        pos);
}


/*
 * Resolve a persistent ledger reference and independently verify the
 * complete BRC checkpoint identity.
 *
 * inode_number is insufficient by itself because the inode number may
 * later be reused.  The stored i_generation must match exactly, after
 * which trusted.brc is cross-checked against the persistent ledger.
 */
static struct inode *
ext4_brc_resolve_checkpoint(
        struct super_block *sb,
        const struct ext4_brc_lineage_header_disk *header,
        const struct ext4_brc_lineage_entry_disk *entry,
        u64 expected_generation)
{
        struct ext4_brc_meta_disk meta;
        struct inode *inode;
        u64 ino;
        int ret;

        if (le32_to_cpu(entry->flags))
                return ERR_PTR(-EFSCORRUPTED);

        ino = le64_to_cpu(entry->inode_number);

        /*
         * Reject a persistent inode number that would be truncated on
         * the current prototype architecture.
         */
        if ((u64)(unsigned long)ino != ino)
                return ERR_PTR(-ESTALE);

        inode = ext4_iget(
                        sb,
                        (unsigned long)ino,
                        EXT4_IGET_HANDLE);
        if (IS_ERR(inode))
                return inode;

        /*
         * Unlike the generic NFS helper, generation zero is not a
         * wildcard.  BRC records an exact inode incarnation.
         */
        if (inode->i_generation !=
            le32_to_cpu(entry->inode_generation)) {
                iput(inode);
                return ERR_PTR(-ESTALE);
        }

        if (!S_ISREG(inode->i_mode) ||
            !IS_IMMUTABLE(inode)) {
                iput(inode);
                return ERR_PTR(-EFSCORRUPTED);
        }

        ret = ext4_brc_read_meta(inode, &meta);
        if (ret <= 0) {
                iput(inode);

                if (ret < 0)
                        return ERR_PTR(ret);

                return ERR_PTR(-EFSCORRUPTED);
        }

        if (le32_to_cpu(meta.magic) !=
            EXT4_BRC_META_MAGIC ||
            le16_to_cpu(meta.version) !=
            EXT4_BRC_META_VERSION ||
            le16_to_cpu(meta.state) !=
            EXT4_BRC_STATE_SEALED ||
            le64_to_cpu(meta.reserved)) {
                iput(inode);
                return ERR_PTR(-EFSCORRUPTED);
        }

        if (memcmp(&meta.lineage_hi,
                   &header->lineage_hi,
                   sizeof(meta.lineage_hi)) ||
            memcmp(&meta.lineage_lo,
                   &header->lineage_lo,
                   sizeof(meta.lineage_lo))) {
                iput(inode);
                return ERR_PTR(-EFSCORRUPTED);
        }

        if (le64_to_cpu(meta.generation) !=
            expected_generation) {
                iput(inode);
                return ERR_PTR(-EFSCORRUPTED);
        }

        return inode;
}


/*
 * Classify physical-block liveness between two verified adjacent
 * SEALED checkpoints:
 *
 *     old = Cg
 *     successor = Cg+1
 *
 * Both checkpoints are already ledger-published.  A BUILDING child
 * can therefore never reach this function as the successor.
 *
 * Classification:
 *
 *   old HOLE
 *       -> OLD_HOLE
 *
 *   old MAPPED(P), successor MAPPED(P)
 *       -> SHARED_LIVE
 *
 *   old MAPPED(P), successor MAPPED(Q), P != Q
 *       -> DEAD_UNIQUE
 *
 *   old MAPPED, successor HOLE
 *       -> EFSCORRUPTED
 *
 * The last case is corruption only because the successor has already
 * completed BRC inheritance, reached SEALED state, and been published
 * into the persistent ledger.  The same observation against a
 * BUILDING child would be a legal intermediate construction state.
 *
 * Traversal follows mapping boundaries, as already done by
 * ext4_brc_inherit_holes(), rather than performing one lookup for
 * every individual logical block.
 */
static int ext4_brc_classify_inode_pair(
        struct inode *old_inode,
        struct inode *successor_inode,
        struct ext4_brc_classify_stats *stats)
{
        struct super_block *sb = old_inode->i_sb;
        ext4_lblk_t lblk = 0;
        ext4_lblk_t total_blocks;
        loff_t size;
        int ret = 0;

        memset(stats, 0, sizeof(*stats));

        if (successor_inode->i_sb != sb)
                return -EXDEV;

        if (!ext4_test_inode_flag(
                    old_inode,
                    EXT4_INODE_EXTENTS) ||
            !ext4_test_inode_flag(
                    successor_inode,
                    EXT4_INODE_EXTENTS))
                return -EOPNOTSUPP;

        /*
         * Current BRC physical lifetime semantics operate at normal
         * ext4 filesystem-block granularity.
         */
        if (EXT4_SB(sb)->s_cluster_ratio != 1)
                return -EOPNOTSUPP;

        size = i_size_read(old_inode);

        /*
         * BRC currently uses the fixed-layout checkpoint invariant.
         */
        if (i_size_read(successor_inode) != size)
                return -EFSCORRUPTED;

        if (!size)
                return 0;

        total_blocks =
                (size + sb->s_blocksize - 1) >>
                sb->s_blocksize_bits;

        /*
         * SEALED checkpoints are immutable.  Hold read locks over
         * both extent trees while taking the classification snapshot.
         */
        down_read(&EXT4_I(old_inode)->i_data_sem);
        down_read(&EXT4_I(successor_inode)->i_data_sem);

        while (lblk < total_blocks) {
                struct ext4_map_blocks old_map = {
                        .m_lblk = lblk,
                        .m_len = total_blocks - lblk,
                };
                struct ext4_map_blocks successor_map = {
                        .m_lblk = lblk,
                        .m_len = total_blocks - lblk,
                };
                unsigned int old_len;
                unsigned int successor_len;
                unsigned int advance;
                int old_mapped;
                int successor_mapped;

                old_mapped = ext4_ext_map_blocks(
                                NULL,
                                old_inode,
                                &old_map,
                                0);
                if (old_mapped < 0) {
                        ret = old_mapped;
                        break;
                }

                successor_mapped = ext4_ext_map_blocks(
                                NULL,
                                successor_inode,
                                &successor_map,
                                0);
                if (successor_mapped < 0) {
                        ret = successor_mapped;
                        break;
                }

                if (old_mapped > 0 &&
                    (old_map.m_flags &
                     EXT4_MAP_UNWRITTEN)) {
                        ret = -EOPNOTSUPP;
                        break;
                }

                if (successor_mapped > 0 &&
                    (successor_map.m_flags &
                     EXT4_MAP_UNWRITTEN)) {
                        ret = -EOPNOTSUPP;
                        break;
                }

                old_len = old_mapped > 0 ?
                          (unsigned int)old_mapped :
                          old_map.m_len;

                successor_len =
                        successor_mapped > 0 ?
                        (unsigned int)successor_mapped :
                        successor_map.m_len;

                if (!old_len || !successor_len) {
                        ret = -EFSCORRUPTED;
                        break;
                }

                advance = min(old_len,
                              successor_len);

                if (!old_mapped) {
                        /*
                         * The retired checkpoint owns no physical
                         * block over this range.  Whether the
                         * successor remains a hole or materializes
                         * private data does not create reclaimable
                         * predecessor storage.
                         */
                        stats->old_hole_blocks += advance;
                } else if (!successor_mapped) {
                        /*
                         * This is illegal only for the verified
                         * SEALED + ledger-published adjacent pair
                         * accepted by the outer classifier.
                         *
                         * A BUILDING successor is intentionally never
                         * admitted here.
                         */
                        ret = -EFSCORRUPTED;
                        break;
                } else if (old_map.m_pblk ==
                           successor_map.m_pblk) {
                        /*
                         * The successor still references exactly the
                         * same predecessor physical storage.
                         */
                        stats->shared_blocks += advance;
                } else {
                        /*
                         * The successor has replaced this logical
                         * range with different physical storage.
                         *
                         * Under linear inheritance, once the direct
                         * successor no longer carries this old pblk,
                         * no downstream checkpoint can reacquire it
                         * by skipping backwards in the lineage.
                         */
                        stats->dead_unique_blocks += advance;
                }

                stats->compared_blocks += advance;
                lblk += advance;
        }

        up_read(&EXT4_I(successor_inode)->i_data_sem);
        up_read(&EXT4_I(old_inode)->i_data_sem);

        if (ret)
                return ret;

        if (stats->shared_blocks +
            stats->dead_unique_blocks +
            stats->old_hole_blocks !=
            stats->compared_blocks)
                return -EFSCORRUPTED;

        if (stats->compared_blocks !=
            (u64)total_blocks)
                return -EFSCORRUPTED;

        return 0;
}


/*
 * Phase 4C read-only adjacent-checkpoint liveness classifier.
 *
 * Persistent lineage state:
 *
 *     [base, reclaim_generation - 1]
 *             already physically reclaimed
 *
 *     [reclaim_generation, head - 1]
 *             logically retired, physically intact/pending
 *
 *     [head, tail]
 *             live
 *
 * Only:
 *
 *     reclaim_generation <= generation < head
 *
 * may be classified.
 *
 * Consequently the old checkpoint can never be the current persistent
 * tail.  Its successor is guaranteed to exist in the ledger and may
 * itself be the tail.
 *
 * This distinction also protects checkpoint construction.  If a live
 * session is currently building C(T+1), the persistent tail CT remains
 * the complete immutable inheritance source for that BUILDING child.
 * CT is outside the reclamation/classification domain as an old
 * checkpoint.
 *
 * Phase 5 will separately handle promotion and final lineage teardown.
 * Tail destruction belongs to that later finalization path and is
 * legal only after the session has ended and no future child can be
 * created.
 */
int ext4_brc_lineage_classify_pair(
        struct file *lineage_file,
        u64 generation,
        struct ext4_brc_classify_stats *stats)
{
        struct inode *ledger_inode =
                file_inode(lineage_file);
        struct super_block *sb =
                ledger_inode->i_sb;
        struct ext4_brc_lineage_header_disk header;
        struct ext4_brc_lineage_entry_disk old_entry;
        struct ext4_brc_lineage_entry_disk successor_entry;
        struct inode *old_inode = NULL;
        struct inode *successor_inode = NULL;
        u64 base;
        u64 reclaim_generation;
        u64 head;
        u64 nr_entries;
        u64 tail;
        u64 index;
        bool successor_is_tail;
        int ret;

        if (!S_ISREG(ledger_inode->i_mode))
                return -EINVAL;

        if (!(lineage_file->f_mode & FMODE_READ))
                return -EBADF;

        if (!stats)
                return -EINVAL;

        /*
         * Snapshot persistent lineage control fields and the two
         * immutable entry references under BRC ledger serialization.
         */
        mutex_lock(&ext4_brc_lineage_io_mutex);

        ret = ext4_brc_file_read_exact(
                        lineage_file,
                        &header,
                        sizeof(header),
                        0);
        if (ret)
                goto out_ledger_unlock;

        ret = ext4_brc_lineage_validate_header_format(
                        &header);
        if (ret)
                goto out_ledger_unlock;

        base = le64_to_cpu(
                        header.base_generation);
        reclaim_generation =
                ext4_brc_lineage_reclaim_generation(
                        &header);
        head = le64_to_cpu(
                        header.head_generation);
        nr_entries = le64_to_cpu(
                        header.nr_entries);

        if (!nr_entries) {
                ret = -ENODATA;
                goto out_ledger_unlock;
        }

        /*
         * Header validation has already proved this derivation does
         * not wrap and that:
         *
         *     base <= head <= tail
         */
        tail = base + nr_entries - 1;

        /*
         * An old checkpoint must have a persistent direct successor.
         *
         * In particular:
         *
         *     generation == tail
         *
         * is a range error, not filesystem corruption.
         *
         * This rule also keeps a tail that is currently parenting a
         * BUILDING child completely outside the reclamation domain.
         */
        if (generation >= tail) {
                ret = -ERANGE;
                goto out_ledger_unlock;
        }

        /*
         * Phase 4C inference is valid only while the retired
         * predecessor still has its original physical mappings.
         *
         * Once generation < R, physical reclamation has already
         * consumed that checkpoint and its remaining extent tree can
         * no longer be treated as the original mapping snapshot.
         */
        if (generation < reclaim_generation ||
            generation >= head) {
                ret = -ERANGE;
                goto out_ledger_unlock;
        }

        index = generation - base;

        ret = ext4_brc_lineage_read_entry(
                        lineage_file,
                        index,
                        &old_entry);
        if (ret)
                goto out_ledger_unlock;

        ret = ext4_brc_lineage_read_entry(
                        lineage_file,
                        index + 1,
                        &successor_entry);
        if (ret)
                goto out_ledger_unlock;

        successor_is_tail =
                generation + 1 == tail;

        mutex_unlock(&ext4_brc_lineage_io_mutex);

        old_inode = ext4_brc_resolve_checkpoint(
                        sb,
                        &header,
                        &old_entry,
                        generation);
        if (IS_ERR(old_inode))
                return PTR_ERR(old_inode);

        successor_inode =
                ext4_brc_resolve_checkpoint(
                        sb,
                        &header,
                        &successor_entry,
                        generation + 1);
        if (IS_ERR(successor_inode)) {
                ret = PTR_ERR(successor_inode);
                successor_inode = NULL;
                goto out_put;
        }

        if (old_inode == successor_inode) {
                ret = -EFSCORRUPTED;
                goto out_put;
        }

        ret = ext4_brc_classify_inode_pair(
                        old_inode,
                        successor_inode,
                        stats);
        if (ret)
                goto out_put;

        ext4_msg(
                sb,
                KERN_INFO,
                "BRC_CLASSIFY: ledger_inode=%lu generation=%llu successor=%llu successor_is_tail=%u old_inode=%lu successor_inode=%lu shared=%llu dead_unique=%llu old_holes=%llu compared=%llu",
                ledger_inode->i_ino,
                (unsigned long long)generation,
                (unsigned long long)(generation + 1),
                successor_is_tail ? 1 : 0,
                old_inode->i_ino,
                successor_inode->i_ino,
                (unsigned long long)
                        stats->shared_blocks,
                (unsigned long long)
                        stats->dead_unique_blocks,
                (unsigned long long)
                        stats->old_hole_blocks,
                (unsigned long long)
                        stats->compared_blocks);

        ret = 0;

out_put:
        if (successor_inode)
                iput(successor_inode);

        if (old_inode)
                iput(old_inode);

        return ret;

out_ledger_unlock:
        mutex_unlock(&ext4_brc_lineage_io_mutex);
        return ret;
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


/*
 * Insert one already-existing physical range into the child's extent tree.
 *
 * The caller must hold child->i_data_sem for write.
 *
 * This operation deliberately does NOT allocate the shared data blocks:
 * @pblk already belongs to a sealed predecessor and therefore remains
 * allocated in the ext4 block bitmap.  Only the child's mapping metadata
 * is added here.
 *
 * i_blocks is nevertheless increased for the child.  BRC presents each
 * checkpoint inode as a complete logical file view even when some of its
 * mapped data blocks are physically shared with another checkpoint.
 */
static int ext4_brc_insert_shared_extent(handle_t *handle,
                                         struct inode *child,
                                         ext4_lblk_t lblk,
                                         ext4_fsblk_t pblk,
                                         unsigned int len)
{
        struct ext4_ext_path *path;
        struct ext4_extent newext = { 0 };
        loff_t bytes;
        int needed;
        int ret;

        if (!len || len > EXT_INIT_MAX_LEN)
                return -EINVAL;

        path = ext4_find_extent(child, lblk, NULL, 0);
        if (IS_ERR(path))
                return PTR_ERR(path);

        /*
         * Use ext4's own credit estimator.  ext4_ext_insert_extent()
         * may need to split the tree and allocate extent metadata,
         * even though BRC allocates no new shared data block.
         */
        needed = ext4_ext_calc_credits_for_single_extent(child,
                                                         len,
                                                         path);

        ret = ext4_datasem_ensure_credits(handle, child,
                                          needed, needed, 0);
        if (ret) {
                ext4_free_ext_path(path);
                return ret;
        }

        newext.ee_block = cpu_to_le32(lblk);
        newext.ee_len = cpu_to_le16(len);
        ext4_ext_store_pblock(&newext, pblk);

        path = ext4_ext_insert_extent(handle, child, path,
                                      &newext, 0);
        if (IS_ERR(path))
                return PTR_ERR(path);

        ext4_free_ext_path(path);

        /*
         * BRC i_blocks semantics:
         *
         * A shared block is still part of the complete block mapping
         * presented by this inode.  Therefore it contributes to this
         * inode's block accounting even though no second physical data
         * block was allocated from the filesystem bitmap.
         *
         * Do not use dquot_alloc_block() here: no physical data-space
         * allocation is taking place in this operation.
         */
        bytes = (loff_t)len << child->i_blkbits;
        inode_add_bytes(child, bytes);

        ret = ext4_mark_inode_dirty(handle, child);
        if (ret)
                return ret;

        /*
         * ext4_ext_insert_extent() modifies the on-disk extent tree
         * directly.  It does not go through ext4_map_create_blocks(),
         * so invalidate any stale HOLE entry in the ES cache.  A later
         * normal lookup will repopulate the cache from the extent tree.
         */
        ext4_es_remove_extent(child, lblk, len);

        return 0;
}


/*
 * Fill the remaining holes of a BUILDING child from its sealed parent.
 *
 * Inheritance is defined by:
 *
 *     mapped(parent) intersection hole(child)
 *
 * Child mappings that already exist are private/dirty mappings and are
 * never overwritten.  Parent holes remain holes in the child.
 *
 * Both inodes are immutable while shared mappings are visible:
 * the parent was sealed previously, and the child is made immutable
 * before this function is entered.
 */
static int ext4_brc_inherit_holes(struct inode *parent,
                                  struct inode *child)
{
        struct super_block *sb = child->i_sb;
        handle_t *handle;
        ext4_lblk_t lblk = 0;
        ext4_lblk_t total_blocks;
        loff_t size;
        u64 inherited_blocks = 0;
        unsigned int inherited_ranges = 0;
        int credits;
        int ret = 0;
        int stop_ret;

        if (parent->i_sb != child->i_sb)
                return -EXDEV;

        if (!ext4_test_inode_flag(parent, EXT4_INODE_EXTENTS) ||
            !ext4_test_inode_flag(child, EXT4_INODE_EXTENTS))
                return -EOPNOTSUPP;

        /*
         * The first BRC remapping prototype intentionally excludes
         * bigalloc.  Shared-block lifetime is defined at filesystem
         * block granularity.
         */
        if (EXT4_SB(sb)->s_cluster_ratio != 1)
                return -EOPNOTSUPP;

        /*
         * BRC currently relies on the fixed-layout checkpoint invariant.
         * Without layout_id, predecessor and child must expose the same
         * logical byte length before inheritance is attempted.
         */
        size = i_size_read(child);
        if (i_size_read(parent) != size)
                return -EINVAL;

        if (!size)
                return 0;

        total_blocks = (size + sb->s_blocksize - 1) >>
                       sb->s_blocksize_bits;

        /*
         * Start with worst-case credits for one extent.  Inside the loop
         * ext4_datasem_ensure_credits() extends/restarts the transaction
         * using the actual child extent path, following ext4 migration's
         * established insertion pattern.
         */
        credits = ext4_ext_calc_credits_for_single_extent(child, 1, NULL);
        handle = ext4_journal_start(child, EXT4_HT_MAP_BLOCKS, credits);
        if (IS_ERR(handle))
                return PTR_ERR(handle);

        if (IS_SYNC(child))
                ext4_handle_sync(handle);

        /*
         * Parent mappings are stable because parent is already sealed
         * and immutable.  Child is also immutable before this helper is
         * called.  The data semaphores protect extent-tree traversal and
         * modification itself.
         */
        down_read(&EXT4_I(parent)->i_data_sem);
        down_write(&EXT4_I(child)->i_data_sem);

        while (lblk < total_blocks) {
                struct ext4_map_blocks cmap = {
                        .m_lblk = lblk,
                        .m_len = total_blocks - lblk,
                };
                struct ext4_map_blocks pmap = {
                        .m_lblk = lblk,
                };
                ext4_lblk_t advance;
                unsigned int inherit_len;
                int mapped;

                /*
                 * Existing child mappings are the blocks materialized
                 * while the checkpoint was BUILDING.  They win over the
                 * predecessor and must never be replaced.
                 */
                mapped = ext4_ext_map_blocks(NULL, child, &cmap, 0);
                if (mapped < 0) {
                        ret = mapped;
                        break;
                }

                if (mapped > 0) {
                        if (cmap.m_flags & EXT4_MAP_UNWRITTEN) {
                                ret = -EOPNOTSUPP;
                                break;
                        }

                        lblk += mapped;
                        continue;
                }

                if (!cmap.m_len) {
                        ret = -EFSCORRUPTED;
                        break;
                }

                /*
                 * Only inspect the parent over the logical range that
                 * is known to be a hole in the child.
                 */
                pmap.m_len = cmap.m_len;

                mapped = ext4_ext_map_blocks(NULL, parent, &pmap, 0);
                if (mapped < 0) {
                        ret = mapped;
                        break;
                }

                if (mapped == 0) {
                        /*
                         * A predecessor hole represents no physical data
                         * block to inherit.  Preserve the hole and move
                         * to the next mapping boundary.
                         */
                        if (!pmap.m_len) {
                                ret = -EFSCORRUPTED;
                                break;
                        }

                        advance = min_t(ext4_lblk_t,
                                        cmap.m_len,
                                        pmap.m_len);
                        lblk += advance;
                        continue;
                }

                if (pmap.m_flags & EXT4_MAP_UNWRITTEN) {
                        ret = -EOPNOTSUPP;
                        break;
                }

                inherit_len = min_t(unsigned int,
                                    mapped,
                                    cmap.m_len);

                ret = ext4_brc_insert_shared_extent(handle,
                                                    child,
                                                    lblk,
                                                    pmap.m_pblk,
                                                    inherit_len);
                if (ret)
                        break;

                inherited_blocks += inherit_len;
                inherited_ranges++;
                lblk += inherit_len;
        }

        up_write(&EXT4_I(child)->i_data_sem);
        up_read(&EXT4_I(parent)->i_data_sem);

        stop_ret = ext4_journal_stop(handle);
        if (!ret)
                ret = stop_ret;

        if (!ret)
                ext4_msg(sb, KERN_INFO,
                         "BRC_INHERIT: parent_inode=%lu child_inode=%lu blocks=%llu ranges=%u",
                         parent->i_ino,
                         child->i_ino,
                         (unsigned long long)inherited_blocks,
                         inherited_ranges);

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
                /*
                 * A marker may already exist when a prior seal reached
                 * the persistent SEALED state but ledger publication
                 * returned an error.  Treat that state as a retryable
                 * publication rather than permanently returning EEXIST.
                 */
                ret = ext4_brc_read_meta(inode, &meta);
                if (ret < 0)
                        goto out_inode;

                if (ret > 0) {
                        if (le16_to_cpu(meta.state) !=
                            EXT4_BRC_STATE_SEALED) {
                                ret = -EEXIST;
                                goto out_inode;
                        }

                        if (!ext4_brc_same_lineage(&meta,
                                                  session)) {
                                ret = -EPERM;
                                goto out_inode;
                        }

                        if (le64_to_cpu(meta.generation) != 0) {
                                ret = -EINVAL;
                                goto out_inode;
                        }

                        /*
                         * Recover the rare partially completed root
                         * transition where the SEALED identity reached
                         * disk but the immutable fence did not.
                         */
                        if (!IS_IMMUTABLE(inode)) {
                                ret = ext4_brc_set_immutable(inode);
                                if (ret)
                                        goto out_inode;
                        }
                } else {
                        if (IS_IMMUTABLE(inode)) {
                                ret = -EPERM;
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
                                goto out_inode;

                        ret = ext4_brc_set_immutable(inode);
                        if (ret) {
                                if (!IS_IMMUTABLE(inode))
                                        ext4_xattr_set(
                                            inode,
                                            EXT4_XATTR_INDEX_TRUSTED,
                                            EXT4_BRC_XATTR_NAME,
                                            NULL, 0,
                                            XATTR_REPLACE);
                                goto out_inode;
                        }
                }

                new_tail = igrab(inode);
                if (!new_tail) {
                        ret = -ESTALE;
                        goto out_inode;
                }

                /*
                 * The checkpoint is now SEALED + immutable, so it is
                 * safe to drop its inode lock before writing the
                 * independent regular-file ledger.  This avoids nested
                 * checkpoint-inode -> ledger-inode write locking.
                 */
                inode_unlock(inode);

                ret = ext4_brc_lineage_append_checkpoint(
                                session, inode, 0);
                if (ret)
                        goto out_unlocked;

                session->tail_inode = new_tail;
                new_tail = NULL;
                session->generation = 0;

                ext4_msg(inode->i_sb, KERN_INFO,
                         "BRC_ROOT_SEAL: inode=%lu generation=0",
                         inode->i_ino);

                ret = 0;
                goto out_unlocked;
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
         * If the checkpoint is already SEALED, a previous attempt
         * reached the safe persistent checkpoint state but failed while
         * publishing lineage membership.  Retry only that publication.
         */
        if (le16_to_cpu(meta.state) ==
            EXT4_BRC_STATE_SEALED) {
                if (!IS_IMMUTABLE(inode)) {
                        ret = -EINVAL;
                        goto out_inode;
                }

                inode_unlock(inode);

                ret = ext4_brc_lineage_append_checkpoint(
                                session,
                                inode,
                                session->generation + 1);
                if (ret)
                        goto out_unlocked;

                goto publish_child;
        }

        if (le16_to_cpu(meta.state) !=
            EXT4_BRC_STATE_BUILDING) {
                ret = -EINVAL;
                goto out_inode;
        }

        /*
         * Establish the seal fence before any shared physical mapping
         * becomes visible.
         *
         * ext4_brc_set_immutable() first waits for direct I/O and writes
         * dirty pages back.  This is required because a delayed-allocation
         * dirty block must become a real child extent before we classify
         * remaining holes as clean/inheritable.
         *
         * If inheritance later fails, the inode deliberately remains
         * BUILDING + immutable.  A retry may continue filling its remaining
         * holes, but userspace cannot modify an already-shared block.
         */
        ret = ext4_brc_set_immutable(inode);
        if (ret)
                goto out_inode;

        ret = ext4_brc_inherit_holes(session->tail_inode, inode);
        if (ret)
                goto out_inode;

        /*
         * Only after the complete logical mapping has been materialized
         * do we publish BUILDING -> SEALED.
         */
        meta.state = cpu_to_le16(EXT4_BRC_STATE_SEALED);

        ret = ext4_brc_write_meta(inode,
                                  &meta,
                                  XATTR_REPLACE);
        if (ret)
                goto out_inode;

        /*
         * The data/mapping state is now SEALED + immutable.  Release
         * this checkpoint inode before writing the independent lineage
         * ledger file.
         */
        inode_unlock(inode);

        ret = ext4_brc_lineage_append_checkpoint(
                        session,
                        inode,
                        session->generation + 1);
        if (ret)
                goto out_unlocked;

publish_child:
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
        goto out_unlocked;

out_inode:
        inode_unlock(inode);

out_unlocked:
        if (old_tail)
                iput(old_tail);

        if (new_tail)
                iput(new_tail);

        mutex_unlock(&session->lock);
        fdput(session_f);
        return ret;

}
