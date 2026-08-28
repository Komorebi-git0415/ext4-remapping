BRC Crash/Restart Validation Plan
=================================

Status
------

Crash testing is intentionally deferred until the end of the experiment.

Kernel:
    6.12.103-brc2-phase4e

The deterministic crash injection facility is compiled in but MUST remain
disabled during ordinary development and experiments:

    /sys/module/ext4/parameters/brc_test_crash_point = 0

Crash points
------------

0
    Disabled.

1
    Crash immediately after one successful destructive rightmost suffix-range
    mutation.  Persistent R has not advanced.

2
    Crash after the complete destructive inode-pair pass, but before
    ext4_force_commit().  Persistent R has not advanced.

3
    Crash after ext4_force_commit() succeeds, but before persistent R is
    written.  This is the primary durability-ordering test.

Safety rules
------------

Use only the dedicated BRC test filesystem:

    /dev/sdb1 mounted at /mnt/brc-test

Never run crash injection against the root filesystem.

Each crash test must create a fresh C0/C1/C2 lineage.

Before enabling a crash point:

    - save all unrelated work
    - verify the test filesystem
    - sync the machine
    - record RUN
    - record R/H/T
    - record FIEMAP
    - record SHA256 of surviving checkpoints
    - record filesystem free-block count

After reboot, inspect persistent state BEFORE retrying reclamation.

Common fresh-lineage setup
--------------------------

Run from:

    cd ~/experiment/brc-tools

Create:

    findmnt /mnt/brc-test || sudo mount /dev/sdb1 /mnt/brc-test

    RUN="/mnt/brc-test/crash-test/run-$(date +%s)"
    LEDGER="$RUN/lineage.brcl"
    C0="$RUN/c0-root.ckpt"
    C1="$RUN/c1-child.ckpt"
    C2="$RUN/c2-child.ckpt"

    echo "RUN=$RUN"

    sudo ./test_brc_lineage_entries create "$RUN"
    sudo ./test_brc_classify_pair scenario "$LEDGER"

    sudo sync

Expected initial cursor:

    B=0 R=0 H=2 T=2

Record baseline:

    stat -f -c 'free_blocks=%f available_blocks=%a' /mnt/brc-test

    stat -c 'inode=%i size=%s blocks512=%b' "$C0"
    sudo filefrag -v "$C0"

    stat -c 'inode=%i size=%s blocks512=%b' "$C1"
    sudo filefrag -v "$C1"
    sha256sum "$C1"

    stat -c 'inode=%i size=%s blocks512=%b' "$C2"
    sudo filefrag -v "$C2"
    sha256sum "$C2"

Common panic setup
------------------

Make panic reboot automatically:

    sudo sysctl -w kernel.panic=5

Verify the injection facility is disabled:

    cat /sys/module/ext4/parameters/brc_test_crash_point

Expected:

    0

Crash Point 3: post-commit / pre-R
---------------------------------

This is the most important crash-consistency experiment and should be run
first when crash testing is eventually enabled.

Enable:

    echo 3 | sudo tee /sys/module/ext4/parameters/brc_test_crash_point

Verify:

    cat /sys/module/ext4/parameters/brc_test_crash_point

Expected:

    3

Trigger:

    sudo ./test_brc_reclaim_physical "$LEDGER"

Expected execution:

    reclaim C0 -> C1
        ->
    destructive physical pass completes
        ->
    ext4_force_commit() succeeds
        ->
    PANIC
        ->
    persistent R has NOT been published

After reboot, reconstruct RUN/LEDGER/C0/C1/C2 from the recorded RUN path.

Verify:

    uname -r
    cat /sys/module/ext4/parameters/brc_test_crash_point

Expected:

    6.12.103-brc2-phase4e
    0

Before retry, expected persistent state:

    R = 0

    C0:
        size remains 16384
        blocks512 = 0
        FIEMAP = no data extents

    C1:
        mappings unchanged
        contents unchanged

    C2:
        mappings unchanged
        contents unchanged

For the standard four-block C0/C1 scenario, exactly one DEAD_UNIQUE data
block should have become free.

Then retry:

    BEFORE=$(stat -f -c %f /mnt/brc-test)

    sudo ./test_brc_reclaim_physical "$LEDGER"

    AFTER=$(stat -f -c %f /mnt/brc-test)

    echo "before=$BEFORE after=$AFTER"

Expected retry classification:

    shared=0
    dead_unique=0
    old_holes=4
    old_R=0
    new_R=1

The retry must not free another data block.

Crash Point 2: destructive pass / pre-force-commit
--------------------------------------------------

Repeat the complete fresh-lineage setup.

Enable:

    echo 2 | sudo tee /sys/module/ext4/parameters/brc_test_crash_point

Trigger:

    sudo ./test_brc_reclaim_physical "$LEDGER"

The kernel panics after the destructive pass but before the explicit
ext4_force_commit().

After reboot:

    R MUST still be 0.

Unlike Crash Point 3, the amount of physical work visible after recovery is
not required to be all-or-nothing.  Individual ext4 transactions may or may
not have become durable before the crash.

The required invariant is that retry safely derives work from the CURRENT
extent state:

    already-consumed range -> old HOLE
    still-present range    -> classify and reclaim normally

After retry:

    R must advance exactly once from 0 to 1.
    C1 and C2 must remain intact.
    No data block may be double-freed.

Crash Point 1: interrupted destructive pass
-------------------------------------------

Repeat the complete fresh-lineage setup.

Enable:

    echo 1 | sudo tee /sys/module/ext4/parameters/brc_test_crash_point

Trigger:

    sudo ./test_brc_reclaim_physical "$LEDGER"

The panic occurs after one successful CURRENT-rightmost suffix-range mutation.

After reboot:

    R MUST still be 0.

The completed range may or may not be durable depending on journal state at
the crash boundary.

Retry must:

    recognize durable completed ranges as old holes,
    process remaining mappings from the CURRENT extent state,
    never double-free a physical block,
    preserve C1 and C2,
    and finally advance R from 0 to 1.

Crash-log collection
--------------------

If the previous boot's kernel log was persisted:

    sudo journalctl -k -b -1 | \
        grep -E 'BRC_TEST_CRASH|EXT4 BRC deterministic|panic'

Absence of the panic log does not invalidate the experiment.  Persistent
ledger state, FIEMAP state, hashes, and allocator state are the primary
evidence.

Final required evidence
-----------------------

The crash suite passes only if all three crash points converge safely after
restart and retry.

The key protocol under test is:

    destructive physical mutation
        ->
    ext4_force_commit()
        ->
    persistent R publication
        ->
    ledger fsync

The strongest Crash Point 3 result is:

    physical state durable = YES
    R advancement durable  = NO

followed by a retry that observes old holes and advances R without performing
a second physical free.
