"""``codeflow evolve``：opt-in Beta self-evolution 命令入口。

``run`` 执行 cold start -> rounds -> unseal，并可从任意 interruption resume；``check`` 只验证
config/model/benchmark setup，不运行 experiment；``status`` 查看 progress，绝不暴露 sealed test
number；``finalize`` 立即结束 run 并执行 one-way unseal。

CLI 只解析参数并委托 :mod:`codeflow.evolver.launch.runner`。命令返回 0 只表示对应 command 完成；
特别是 check/status 不代表 candidate 有正向 evidence，finalize 也不保证 sealed result 为正。
"""

from __future__ import annotations

import argparse
import sys

from codeflow.product import CLI_NAME


def build_parser() -> argparse.ArgumentParser:
    """构造 ``codeflow evolve`` 的 argparse command tree。

    四个 subcommand 都要求 ``--config``，并支持 ``--smoke``；run 额外提供 ``--force``，
    finalize 额外要求 ``--yes`` 才执行 one-way 操作。函数只创建 parser，不读取配置。
    """
    p = argparse.ArgumentParser(
        prog=f"{CLI_NAME} evolve",
        description=(
            "Run the opt-in Evolver Beta on a registered benchmark. Candidates require manual activation by default."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", required=True, help="run spec YAML")
        sp.add_argument("--smoke", action="store_true", help="shrunk verification run in <work_dir>_smoke")

    run = sub.add_parser("run", help="start or resume an evolution run")
    common(run)
    run.add_argument("--force", action="store_true", help="override the unseal / config-drift guards")

    check = sub.add_parser("check", help="validate config/models/bench setup, run nothing")
    common(check)

    status = sub.add_parser("status", help="show run progress (sealed-safe)")
    common(status)

    fin = sub.add_parser("finalize", help="terminate now and unseal (one-way)")
    common(fin)
    fin.add_argument("--yes", action="store_true", help="confirm ending the run")

    return p


def main(argv: list[str] | None = None) -> int:
    """解析 ``argv`` 并分派 run/check/status/finalize。

    ``argv=None`` 时使用 process args。runner command 的 integer exit code 原样返回；理论上的
    unknown command 返回 2，但 argparse 已要求 subcommand。
    """
    args = build_parser().parse_args(argv)
    from codeflow.evolver.launch import runner

    if args.command == "run":
        return runner.cmd_run(args.config, smoke=args.smoke, force=args.force)
    if args.command == "check":
        return runner.cmd_check(args.config, smoke=args.smoke)
    if args.command == "status":
        return runner.cmd_status(args.config, smoke=args.smoke)
    if args.command == "finalize":
        return runner.cmd_finalize(args.config, smoke=args.smoke, yes=args.yes)
    return 2


if __name__ == "__main__":
    sys.exit(main())
