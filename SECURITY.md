# Security / 安全

Sparse-DFT is research software for local workstations. It has no network service and no authentication layer.

Several components handle files or executables that come from outside:
- The GUI and the worker run native executables named in user settings.
- The program reads user-supplied structure files.
- It also reads PAW archives and job JSON files.

Treat these inputs as untrusted unless they come from you. Do not expose a job directory or a worker to other users.

Report a suspected vulnerability privately to the maintainers through the repository's security advisory page. Do not put working exploits in a public issue.

If you believe a file in Sparse-DFT contains material that its licence does not allow, report it the same way. This covers third-party code, text or data. Name the file and lines and the source you suspect. The maintainers will check the report and remove or replace any confirmed material.

---

Sparse-DFT 是供本地工作站使用的科研软件，没有网络服务，也没有身份认证层。

以下组件会处理来自外部的文件或可执行程序：
- GUI 和 worker 会运行用户设置中指定的原生程序。
- 程序会读取用户提供的结构文件。
- 程序还会读取 PAW 归档和作业 JSON 文件。

除非这些输入来自你本人，否则请视为不可信。不要把作业目录或 worker 暴露给其他用户。

如发现疑似漏洞，请通过仓库的安全公告页面私下报告维护者，不要在公开 issue 中贴出可利用的攻击代码。

如果你认为 Sparse-DFT 的某个文件含有其许可不允许的第三方代码、文字或数据，也请用同样方式报告，并写明文件、行号和你怀疑的来源。维护者核实后会删除或替换确认有问题的内容。
