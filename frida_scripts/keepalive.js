const libc = "libsystem_kernel.dylib";
["exit", "_exit", "abort", "pthread_kill", "kill"].forEach(s => {
    const p = Module.findGlobalExportByName(s);
    if (p) Interceptor.attach(p, { onEnter() { console.log("⚠️ " + s + " called"); } });
});
