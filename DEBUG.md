# Debugging and multi-thread

Using `rust-gdb` is the best way to debug deadlocks in PyO3 projects because it includes "pretty printers" that make Rust types (like `Vec`, `String`, and `HashMap`) readable in the debugger. 

1. Installation
`rust-gdb` is a wrapper script for `gdb`. You must have both the standard GDB and the Rust toolchain installed.
Linux/WSL: Most distributions include GDB. If not, install it via your package manager (e.g., `sudo apt install gdb`). Rustup installs rust-gdb automatically alongside rustc.

2. Preparing Your Build
For the debugger to show your source code and variable names, you must build with debug symbols.
During Development: Run maturin develop. This builds in debug mode by default.
In Release: If you must debug a release build, ensure your Cargo.toml includes debug = true under [profile.release]. 

3. Basic Operation: Launching with Python
To debug a Rust library called from Python, you actually run the Python interpreter inside `rust-gdb`. 
Start the debugger:
```bash
rust-gdb --args python your_script.py
```

Set a breakpoint:
Inside the GDB prompt, tell it where to stop in your Rust code.
```gdb
(gdb) break your_filename.rs:line_number
```

Run the program:
```gdb
(gdb) run
```

4. Special Case: Debugging a Hang (Attach to Process)
Since your process is getting stuck, it is often easier to attach to it while it is hanging. 
Find the Process ID (PID) of your hanging Python script:
```bash
ps aux | grep python
```

Attach rust-gdb:
```bash
rust-gdb -p <PID>
```

Inspect the threads:
The program will pause immediately upon attaching. To see where your parallel threads are stuck, use:
```gdb
(gdb) thread apply all bt
```

Look for any thread waiting on a "futex" or "lock"—this is where your deadlock is occurring.
5. Essential GDB Commands
```
c (continue): Resume execution until the next breakpoint or crash.
n (next): Step over the current line of code.
s (step): Step into a function.
p <variable> (print): Show the value of a Rust variable.
l (list): Show the source code around the current execution point.
q (quit): Exit the debugger. 
```