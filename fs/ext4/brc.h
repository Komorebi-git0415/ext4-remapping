/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _EXT4_BRC_H
#define _EXT4_BRC_H

struct inode;
struct file;

struct ext4_brc_classify_stats {
        u64 shared_blocks;
        u64 dead_unique_blocks;
        u64 old_hole_blocks;
        u64 compared_blocks;
};

int ext4_brc_lineage_classify_pair(
        struct file *lineage_file,
        u64 generation,
        struct ext4_brc_classify_stats *stats);

int ext4_brc_lineage_begin(struct file *lineage_file);
int ext4_brc_lineage_reclaim_through(struct file *lineage_file,
                                     u64 through_generation);
int ext4_brc_session_begin(struct file *anchor_file);
int ext4_brc_prepare_child(struct file *child_file,
                           struct file *parent_file,
                           int session_fd);
int ext4_brc_seal_with_session(struct file *file,
                               int session_fd);
int ext4_brc_has_marker(struct inode *inode);
int ext4_brc_seal_inode(struct inode *inode);

#endif /* _EXT4_BRC_H */
