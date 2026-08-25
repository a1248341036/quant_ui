$ErrorActionPreference = 'Stop'

$source = @'
using System;
using System.Text;
using System.Runtime.InteropServices;

public static class RemoteProcessEnvironment {
    [StructLayout(LayoutKind.Sequential)]
    struct ProcessBasicInformation {
        public int ExitStatus;
        public int Padding;
        public IntPtr Peb;
        public IntPtr A;
        public IntPtr B;
        public IntPtr Unique;
        public IntPtr C;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern IntPtr OpenProcess(uint access, bool inherit, int pid);

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool ReadProcessMemory(IntPtr process, IntPtr address, byte[] buffer, int size, out IntPtr read);

    [DllImport("ntdll.dll")]
    static extern int NtQueryInformationProcess(IntPtr process, int infoClass, IntPtr info, int length, out int returned);

    static IntPtr ReadPointer(IntPtr process, IntPtr address) {
        byte[] buffer = new byte[IntPtr.Size];
        IntPtr read;
        if (!ReadProcessMemory(process, address, buffer, buffer.Length, out read) || read.ToInt64() != buffer.Length) {
            throw new System.ComponentModel.Win32Exception();
        }
        return new IntPtr(BitConverter.ToInt64(buffer, 0));
    }

    public static string ReadEnvironment(int pid) {
        IntPtr process = OpenProcess(0x410, false, pid);
        if (process == IntPtr.Zero) throw new System.ComponentModel.Win32Exception();
        IntPtr info = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(ProcessBasicInformation)));
        try {
            int returned;
            int status = NtQueryInformationProcess(process, 0, info, Marshal.SizeOf(typeof(ProcessBasicInformation)), out returned);
            if (status != 0) throw new Exception("NtQueryInformationProcess: " + status);
            ProcessBasicInformation pbi = (ProcessBasicInformation)Marshal.PtrToStructure(info, typeof(ProcessBasicInformation));
            IntPtr processParameters = ReadPointer(process, IntPtr.Add(pbi.Peb, 0x20));
            IntPtr environment = ReadPointer(process, IntPtr.Add(processParameters, 0x80));
            byte[] buffer = new byte[1024 * 1024];
            IntPtr read;
            if (!ReadProcessMemory(process, environment, buffer, buffer.Length, out read)) throw new System.ComponentModel.Win32Exception();
            return Encoding.Unicode.GetString(buffer, 0, read.ToInt32()).TrimEnd('\0');
        } finally {
            Marshal.FreeHGlobal(info);
        }
    }
}
'@

Add-Type -TypeDefinition $source

$targets = @(Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -eq 'TeleAgent.exe' -and $_.CommandLine -match '(?i)TeleAgent') -or
    ($_.Name -eq 'node.exe' -and $_.CommandLine -match '(?i)TeleAgent')
})

foreach ($target in $targets) {
    try {
        $envText = [RemoteProcessEnvironment]::ReadEnvironment([int]$target.ProcessId)
        $matches = @($envText -split "`0" | Where-Object {
            $_ -match '(?i)(TOKEN|AUTH|SUPER_AGENT|OPENCODE_SERVER_PASSWORD|API_KEY|API_BASE)'
        } | ForEach-Object {
            $parts = $_ -split '=', 2
            [ordered]@{ name = $parts[0]; length = if ($parts.Count -eq 2) { $parts[1].Length } else { 0 } }
        })
        [ordered]@{ pid = $target.ProcessId; name = $target.Name; vars = $matches } | ConvertTo-Json -Compress -Depth 5
    } catch {
        [ordered]@{ pid = $target.ProcessId; name = $target.Name; error = $_.Exception.Message } | ConvertTo-Json -Compress
    }
}
