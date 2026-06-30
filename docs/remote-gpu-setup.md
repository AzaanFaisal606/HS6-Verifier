# Remote GPU Server — Connect, Mount, Run the VLM

Goal: keep working from this machine (where Claude Code runs) without installing
anything Claude-related on the GPU server. We **mount** the server's filesystem
here with `sshfs` so files can be edited locally, and **run** GPU code on the
server over plain `ssh`.

> **Key distinction.** `sshfs` mounts the *filesystem* — it lets you read/write
> the server's files as if they were local. It does **not** run code on the
> server. The VLM needs the server's GPUs, so the *script executes server-side*
> via `ssh` (or an interactive ssh session). Editing = local (over the mount);
> execution = remote (over ssh). Both use the same SSH connection.

Placeholders used throughout — substitute your own:

| placeholder | meaning | example |
|---|---|---|
| `USER` | your login on the GPU server | `azaan` |
| `HOST` | server hostname or IP | `10.0.0.42` or `gpubox.lan` |
| `PORT` | SSH port (omit if 22) | `22` |
| `REMOTE_DIR` | working dir on the server | `/home/USER/ptc-corpus` |

Auth here is **password-based** (no key set up). `sshfs` and `ssh` will prompt
for the password interactively. Because Claude Code cannot type an interactive
password, **you run the mount + first ssh yourself** — in the Claude Code prompt,
prefix a command with `!` to run it in this session so its output is captured,
e.g. `! sshfs ...`.

> Tip: to avoid retyping the password on every command, start one persistent SSH
> control-master connection (see "Optional: connection multiplexing" below) — you
> authenticate once and every later `ssh`/`sshfs` reuses it without a prompt.

---

## 0. Prerequisites (already satisfied on this machine)

- `sshfs` → `/usr/bin/sshfs` (installed)
- `ssh`   → `/usr/bin/ssh`
- `fusermount3` (FUSE 3) → present
- Distro: Fedora 43 (`dnf` if anything else is ever needed)

Nothing to install. If `sshfs` were ever missing: `sudo dnf install sshfs`.

---

## 1. Sanity-check the SSH connection

```bash
ssh -p PORT USER@HOST 'hostname; nvidia-smi -L'
```

- Confirms login works (enter password when prompted).
- `nvidia-smi -L` lists the GPUs — confirms you're on the GPU box and the driver
  is up. If this fails, fix SSH/GPU access before mounting.

---

## 2. Create a local mountpoint

```bash
mkdir -p "/home/azaan/Documents/PTC Corpus/remote-gpu"
```

This empty dir is where the server's `REMOTE_DIR` will appear.

---

## 3. Mount the server dir here with sshfs

Run this yourself (password prompt):

```bash
! sshfs -p PORT USER@HOST:REMOTE_DIR "/home/azaan/Documents/PTC Corpus/remote-gpu" \
    -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,follow_symlinks
```

- `reconnect` + `ServerAliveInterval` → survive brief network blips.
- After this, `remote-gpu/` shows the server's files. Edit them with normal local
  tools; writes go straight to the server.

Verify:

```bash
ls -la "/home/azaan/Documents/PTC Corpus/remote-gpu"
mountpoint "/home/azaan/Documents/PTC Corpus/remote-gpu"   # -> "is a mountpoint"
```

---

## 4. Push this project up to the server

Mount exposes the server dir locally, so a plain `cp` (or `rsync` over the mount)
copies the project onto the GPU box. Send only what the VLM step needs — the DB,
the scripts, and the chapter docs:

```bash
DST="/home/azaan/Documents/PTC Corpus/remote-gpu"
cp "/home/azaan/Documents/PTC Corpus/pct/pct_corpus.db"        "$DST"/
cp "/home/azaan/Documents/PTC Corpus/pct/build_embeddings.py"  "$DST"/
cp "/home/azaan/Documents/PTC Corpus/vlm_infer.py"             "$DST"/   # the VLM script
cp -r "/home/azaan/Documents/PTC Corpus/docs"                  "$DST"/
```

(Or, faster for large/again-and-again syncs, rsync directly over ssh — bypasses
the FUSE layer:)

```bash
rsync -avP -e "ssh -p PORT" \
  "/home/azaan/Documents/PTC Corpus/pct/pct_corpus.db" \
  "/home/azaan/Documents/PTC Corpus/vlm_infer.py" \
  USER@HOST:REMOTE_DIR/
```

---

## 5. Set up the Python env on the server (one-time)

GPU libraries (torch+CUDA, the VLM weights) must live **on the server**. Do this
over ssh, not the mount:

```bash
ssh -p PORT USER@HOST
# --- now on the server ---
cd REMOTE_DIR
python3 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match server CUDA
pip install transformers accelerate pillow sqlite-vec sentence-transformers
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
exit
```

Pin the CUDA wheel (`cu121`/`cu124`/…) to whatever `nvidia-smi` reports on the
server.

---

## 6. Run the VLM on the server

Two equally fine options:

**A. One-shot over ssh** (good for a single run, output streams back here):

```bash
ssh -p PORT USER@HOST 'cd REMOTE_DIR && . .venv/bin/activate && python vlm_infer.py --image some.png'
```

**B. Interactive session** (good for iterating / long runs):

```bash
! ssh -p PORT USER@HOST
# then on the server:
cd REMOTE_DIR && . .venv/bin/activate
python vlm_infer.py ...
```

For long jobs that must survive disconnects, wrap in `tmux`/`nohup` on the server:

```bash
ssh -p PORT USER@HOST 'cd REMOTE_DIR && . .venv/bin/activate && nohup python vlm_infer.py > vlm.log 2>&1 &'
# watch progress via the mount:
tail -f "/home/azaan/Documents/PTC Corpus/remote-gpu/vlm.log"
```

Because the project is mounted, outputs the script writes into `REMOTE_DIR`
(e.g. an updated `pct_corpus.db`, result JSON) appear immediately under
`remote-gpu/` here — no separate download step.

---

## 7. Unmount when done

```bash
fusermount3 -u "/home/azaan/Documents/PTC Corpus/remote-gpu"
```

---

## Optional: connection multiplexing (type password once)

Add to `~/.ssh/config` so all connections to the server share one authenticated
socket — you enter the password only for the first connection:

```
Host gpu
    HostName HOST
    User USER
    Port PORT
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m
    ServerAliveInterval 15
```

Then everything shortens to `ssh gpu`, `sshfs gpu:REMOTE_DIR remote-gpu`, etc.,
and only the first prompts for a password.

> If you later want passwordless auth entirely: `ssh-keygen -t ed25519` then
> `ssh-copy-id -p PORT USER@HOST`. After that Claude Code can run `ssh`/`sshfs`
> directly without you typing anything.

---

## Notes / gotchas

- **sshfs ≠ remote shell.** Editing files over the mount is local I/O; running
  `python vlm_infer.py` *must* go through `ssh` to use the server GPUs.
- **Don't run GPU code through the mount path locally** — `python remote-gpu/vlm_infer.py`
  executes on *this* machine (no GPU) reading files over the network. Always
  `ssh` into the server to run.
- **Heavy files over FUSE are slow.** For the DB and model weights prefer `rsync`
  over `ssh` (step 4) instead of `cp` over the mount.
- **VLM weights live server-side** in the HF cache (`~/.cache/huggingface`),
  downloaded on first run on the server — not over the mount.
