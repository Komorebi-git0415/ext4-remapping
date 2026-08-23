/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _EXT4_BRC_H
#define _EXT4_BRC_H

struct inode;

int ext4_brc_has_marker(struct inode *inode);
int ext4_brc_seal_inode(struct inode *inode);

#endif /* _EXT4_BRC_H */
