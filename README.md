# KingOfGpu 使用说明

KingOfGpu 会寻找剩余显存至少 30 GiB 的 GPU，用本项目自己的 CUDA 进程占住可用显存，并通过飞书通知你 GPU 编号。你收到通知后，使用本机命令释放 KingOfGpu 的占用器，再启动自己的真实代码。

项目不会停止、暂停或修改服务器上其他用户的程序。它只会停止自己启动的 `kingofgpu.occupier` 进程。

监控器本身可以使用 base Python 运行；占用器使用配置中的 CUDA Python 环境。当前服务器默认配置为 `/home/xujunyi/anaconda3/envs/dubins/bin/python`，该环境已验证可用 PyTorch 和 CUDA。

## 一、首次配置飞书

飞书自定义机器人只能向所在群发送消息，不能接收命令，因此释放操作使用服务器终端执行。官方文档：

- https://open.feishu.cn/document/ukTMukTMukTM/ucTM5YjL3ETO24yNxkjN

在飞书群中添加“自定义机器人”，开启“签名校验”，复制 Webhook URL 和 Secret，然后编辑配置：

```bash
cd /home/xujunyi/KingOfGpu
vim config.json
```

填写：

```json
"webhook_url": "你的飞书Webhook地址",
"secret": "你的签名Secret"
```

测试通知：

```bash
python3 -m kingofgpu test-notify
```

收到“飞书通知测试成功”后再启动监控器。

如果更换服务器或 CUDA 环境，只需修改 `config.json` 中的 `occupier_python`，并确认该解释器可以执行：

```bash
/home/xujunyi/anaconda3/envs/dubins/bin/python -c "import torch; print(torch.cuda.is_available())"
```

## 二、启动监控器

推荐在 tmux 中长期运行：

```bash
cd /home/xujunyi/KingOfGpu
tmux new -s kingofgpu
./run_tmux.sh
```

如果只想在本次运行中最多占用两张 GPU，不修改 `config.json`：

```bash
./run_tmux.sh --max_gpus 2
```

也支持连字符写法 `--max-gpus 2`。命令行参数只对本次监控器运行生效；不传参数时继续使用 `config.json` 中的 `max_gpus`。

按 `Ctrl-b`，再按 `d`，可以退出 tmux 但保持程序运行。

重新进入：

```bash
tmux attach -t kingofgpu
```

监控器启动后会发送一条飞书消息，说明候选条件和释放命令。没有符合条件的 GPU 时不会启动占用器，只会继续等待。

查看实时 GPU 状态：

```bash
cd /home/xujunyi/KingOfGpu
python3 -m kingofgpu status
```

实验调度时，应先用带预算的状态查询，而不是只看 `nvidia-smi` 的空闲列：

```bash
python3 -m kingofgpu status --required-mib 24000
python3 -m kingofgpu status --required-mib 24000 --json
```

状态会分别报告 `non_kog_processes`（非 KingOfGpu 进程的逐 PID 显存和）、
`kog`（本项目预留）、`unattributed`（whole-GPU 计数中未被逐 PID 解释的部分）与
`free_after_kog_release`（仅释放本项目预留后的可用显存）以及对预算的容纳判断。
这使“当前被预留”与“真实被其他任务占用”可区分。`--json` 还给出 PID、OS 用户、
命令行和用户任务绑定状态，便于实验记录程序保存一次可复核的调度快照。

## 三、收到 GPU 通知后如何释放

飞书消息中会给出具体 GPU 编号，例如 GPU 2。请在服务器另开一个终端执行：

```bash
cd /home/xujunyi/KingOfGpu
python3 -m kingofgpu release --gpu 2
```

如果项目同时占用了多张 GPU，只释放某一张：

```bash
python3 -m kingofgpu release --gpu 2
```

释放全部由 KingOfGpu 占用的 GPU：

```bash
python3 -m kingofgpu release --all
```

释放命令只会停止 KingOfGpu 自己启动的占用器，不会停止其他用户的训练、推理或服务程序。

执行 `release` 后不需要重启监控器。该 GPU 会进入低频监控状态：默认每 60 秒检查一次，而不是跟随全局的正常轮询频率。只要它仍被你的程序使用，就不会被重新占用；当它再次满足候选条件时，监控器会重新占用它，并恢复该 GPU 的正常监控状态。

## 四、释放后启动自己的代码

应先根据预算选择一张卡；若该卡只是由 KingOfGpu 预留，释放**选中的这一张**：

```bash
cd /home/xujunyi/KingOfGpu
python3 -m kingofgpu release --gpu 2
```

随后立即启动用户任务。`release --gpu N` 会建立 `xujunyi` 任务绑定；不要在选卡后
反复执行 `release --all` 或启动多个 `--max-gpus 2` 监控器，否则会制造无意义的显存
竞争。`release --all`只用于清理监控预留或一次性的观测前准备，不会建立绑定。

如需人工核对，再执行：

```bash
nvidia-smi
```

然后在新的 tmux 会话中启动真实代码，例如：

```bash
tmux new -s my-job
cd /你的项目目录
python your_train.py
```

如果自己的程序也需要长期运行，按 `Ctrl-b`、`d` 脱离 tmux。此时不要关闭 `kingofgpu` 监控会话，它会继续寻找下一张符合条件的 GPU。

## 跨项目任务完成飞书通知

在任何项目中，可用 `notify-run` 包装长时间运行的流水线。它不会把完整命令行或参数发送到飞书，因此不会意外暴露令牌等敏感参数；飞书消息只包含任务名、结果、耗时与退出码。

```bash
tmux new-session -d -s my-training \
  'cd /你的项目目录 && mkdir -p logs && \
  python3 -m kingofgpu --config /home/xujunyi/KingOfGpu/config.json \
    notify-run --name my-training -- bash pipeline.sh \
    > logs/my-training.log 2>&1'
```

`--config` 必须显式指向 KingOfGpu 的私密 `config.json`；不要将 webhook 或 secret 复制到其他项目。`--name` 可省略，此时通知会使用脚本名或命令名。

任务以原样参数启动，标准输出和错误输出会保留在当前终端（上例重定向至日志文件）。零退出码会发送“完成”通知，非零退出码会发送“失败”通知；tmux 会话中的任务收到 `SIGINT` 或 `SIGTERM` 时，会转发信号给任务并发送“中断”通知。飞书网络故障只会写入标准错误，绝不会覆盖原任务退出码。

## 五、再次启动和停止

服务器重启或监控器被手动停止后，再次启动：

```bash
cd /home/xujunyi/KingOfGpu
tmux new -s kingofgpu
./run_tmux.sh
```

停止 KingOfGpu 监控器：

```bash
tmux attach -t kingofgpu
```

在窗口中按 `Ctrl-C`。监控器退出时会释放它自己启动的占用器，但不会触碰其他程序。

## 六、配置多 GPU

默认最多占用一张 GPU：

```json
"max_gpus": 1
```

例如最多占用两张：

```json
"max_gpus": 2
```

如果不希望修改配置文件，也可以在启动时临时覆盖：

```bash
./run_tmux.sh --max_gpus 2
```

其中 `--max_gpus 0` 表示不限制占用数量。

释放时始终使用实际 GPU 编号，例如 `release --gpu 5`。

## 重要说明

项目不再根据“原有程序退出”自动释放 GPU。正确流程是：

```text
收到飞书通知
→ 在服务器终端执行 release --gpu 编号
→ 用 nvidia-smi 确认显存释放
→ 启动自己的真实代码
```

`release --gpu N` 是人工交接命令：它会在 `bind_wait_seconds` 窗口内等待 OS 用户 `release_bind_user`（默认`xujunyi`）的 GPU 任务。一旦检测到该用户的 PID，monitor 将绑定该 PID，直至 PID 退出前都不会重新占用这张 GPU；PID 全部退出后，监控器会在下一轮轮询立即重新评估并可再次占用该 GPU，不再等待原先的冷却时间。`release --all`仅释放本项目占用器，不创建用户任务绑定，适合清理监控预留。

这一绑定避免“真实任务仍在运行但显存仍高于候选阈值”时 monitor 重占同一张卡。它只认配置的本地 OS 用户；其他用户的新 GPU 进程不会触发占用器自动释放。

配置中的 `release_on_new_process` 默认仍为开启状态：如果占用器运行期间确实出现了新的外部 GPU 进程，项目会释放自己的占用器。但由于占用器占用了大部分显存，新程序可能因显存不足而无法成功初始化，因此推荐始终使用明确的 `release --gpu 编号` 命令。

如果占用器启动失败，监控器只向飞书发送一次失败通知，并停止在本次运行中重复尝试，避免错误消息和重试刷屏。修复 `occupier_python` 后重启监控器即可。
