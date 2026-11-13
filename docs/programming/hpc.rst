HPC engineering
===============

Environment
-----------

Slurm with an interactive partition capped at **12 h**. GPUs are NVIDIA
L40S cards split by MIG into 6, 12, or 24 GB slices, and no full card is
available.

Checkpoint chaining
-------------------

A job that needs :math:`H` hours on a partition capped at :math:`h` hours is
split into :math:`\lceil H/h\rceil` hops:

.. math::

   H_\text{fold} = 1000 \times 168\ \text{s} \approx 47\ \text{h}
   \;\Rightarrow\; \lceil 47/12 \rceil = 4\ \text{hops per fold}.

Each hop is an ``sbatch`` with ``--dependency=afterany:<previous>`` and
resumes from the latest checkpoint. That gives three requirements:

* **Idempotent workers.** A hop on a finished fold must exit cleanly.
* **Clean job IDs on stdout.** A chaining helper that echoed progress lines
  to stdout made the captured "job ID" contain text, and Slurm rejected it
  with ``Job dependency problem``. Logging goes to stderr, and only the ID goes
  to stdout:

  .. code-block:: bash

     jid=$(sbatch --parsable ... "$worker")   # ID only
     echo "hop=${hop} job=${jid}" >&2          # human log on stderr
     printf '%s\n' "${jid}"                    # value for the caller

* **Explicit walltime per hop.** A helper default of 4 h made SynthSeg v2
  hops time out. Asking for 24 h on the 12 h partition kept a job pending
  forever with ``PartitionTimeLimit``.

Batch scripts run from a copy
-----------------------------

``sbatch`` copies the batch script into the Slurm spool directory before
running it. ``$(dirname "${BASH_SOURCE[0]}")`` then points at the spool,
not the repository, so helper files are not found. Workers resolve the
project root from ``SLURM_SUBMIT_DIR`` instead:

.. literalinclude:: ../../code/synthseg_pc/train_job.sh
   :language: bash
   :lines: 1-8

Submit helpers run on the login node, so they can still use ``BASH_SOURCE``.

Shared filesystem
-----------------

Recursive ``find`` over NFS and some ``squeue``/``sinfo`` calls hung for
minutes. Queries are kept narrow and bounded by timeouts.

Dry-run by default
------------------

Multi-job orchestration scripts print the full submission plan and queue
nothing unless ``RUN=1`` is set. That lets a reviewer read exactly what
would be launched before any GPU time is spent.
