
## 1. operating system

- Ubuntu Server LTS 또는 RHEL + 최신 GPU 하드웨어 지원 커널
- 드라이버가 생성하는 디바이스 파일

| 파일                    | 역할                     |
| --------------------- | ---------------------- |
| `/dev/nvidia0,1,2...` | GPU 하나당 하나             |
| `/dev/nvidiactl`      | 드라이버 제어 연산             |
| `/dev/nvidia-uvm`     | Unified Virtual Memory |
| `/dev/nvidia-modeset` | mode-setting, 버퍼 관리    |

## 2. NVIDIA Software stack

```shell
┌─────────────────────────────────────────┐
│ Frameworks   PyTorch / TensorFlow / JAX │
├─────────────────────────────────────────┤
│ Libraries    cuDNN, cuBLAS, NCCL,       │
│              CUTLASS, Triton, Warp      │
├─────────────────────────────────────────┤
│ Compiler     nvcc / CUDA C++            │
├─────────────────────────────────────────┤
│ Runtime      CUDA Runtime (cudart)      │
├─────────────────────────────────────────┤
│ Driver       NVIDIA GPU Driver          │
├─────────────────────────────────────────┤
│ Hardware     GPU                        │
└─────────────────────────────────────────┘
```

2.1 GPU Driver
- Linux OS ↔ GPU 하드웨어 인터페이스

2.2 CUDA Toolkit & Runtime
- nvcc : CUDA C++ kernel compiler
- cudart : 컴파일된 프로그램이 링크하는 런타임. 드라이버와 직접 통신해 작업 실행·메모리 할당
- Optim library : cuDNN, cuBLAS(선형대수), NCCL(멀티 GPU 통신) 등


## 3. Configuring the CPUs and OS for GPU Environments

- GPU가 완전한 활용률에 도달하지 못하는 가장 흔한 원인은 CPU가 GPU에 유용한 작업을 계속 공급하지 못하기 때문

1. **CPU affinity 설정** — cross-NUMA 트래픽 회피, 맞는 코어가 맞는 데이터를 처리
2. **메모리 할당 전략** — NUMA 페널티 회피
3. **OS 레벨 변경**

여기에 더해 백그라운드 데몬과 OS 작업을 별도 코어로 격리 — GPU에 데이터를 먹이는 코어에서 떼어냄

## 4. NUMA

- NUMA(Non-Uniform Memory Access) : CPU, GPU, NIC, 메모리가 물리적으로 가까이 묶인 그룹
- 단일 NUMA 노드 안에서의 자원 접근이 다른 노드 접근보다 빠름
- 우리 환경 H200 NVL
> 노드에 GPU 8장, GPU 0–3이 NUMA node 0에, GPU 4–7이 NUMA node 1에 연결

```
        ┌─ NUMA 0 ─────┐              ┌─ NUMA 1 ─────┐
        │  GPU 0 1 2 3 │              │  GPU 4 5 6 7 │
        │   NVLink     │              │   NVLink     │
        │   132 GB/s   │              │   132 GB/s   │
        │              │              │              │
        │  CPU cores   │◀─── UPI ────▶│  CPU cores   │
        │  local DRAM  │   38.5 GB/s  │  local DRAM  │
        └──────────────┘              └──────────────┘
```

GPU 4에 데이터를 먹이려는 CPU 프로세스는 NUMA node 1의 코어에서 돌아야 함

```shell
numactl --hardware
```


## 5. GPU serving system tuning

1. THP(Transparent huge pages)
- THP의 백그라운드 compaction이 예측 불가능한 정지를 유발하고, latency에 치명적
- decoding은 매 토큰이 사용자에게 보이는 latency기 때문에 몇ms stall 하나가 ITL 증가 요소

```
할당 요청 → 2MB 연속 영역 없음 → 커널이 페이지 이동으로 조각모음
                                    ▲ 이 동안 프로세스 정지 (수 ms ~ 수십 ms)
```

```bash
# 현재 설정
cat /sys/kernel/mm/transparent_hugepage/enabled
cat /sys/kernel/mm/transparent_hugepage/defrag

# vLLM이 THP를 실제로 쓰나 (0이면 꺼도 무의미)
grep AnonHugePages /proc/$(pgrep -f vllm | head -1)/smaps_rollup

# compaction이 실제로 발생하나  ← 이게 핵심 판정
grep -E "compact_stall|compact_fail|thp_fault_fallback" /proc/vmstat
sleep 300
grep -E "compact_stall|compact_fail|thp_fault_fallback" /proc/vmstat
# 5분 간격으로 compact_stall이 증가하면 확정
```


```bash
# 런타임 (재부팅 시 초기화)
echo never > /sys/kernel/mm/transparent_hugepage/defrag

# 부족하면 한 단계 더
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled

# 영구 — GRUB
GRUB_CMDLINE_LINUX="... transparent_hugepage=madvise"
update-grub && reboot
```

Kubernetes 노드면 DaemonSet + initContainer(privileged)로 자동화 가능
-> `compact_stall` 증가가 멈추고 p99 ITL 분산이 줄면 OK



2. nvidia-persistenced
- GPU를 아무도 안 쓰면 드라이버가 저전력 상태로 내리고 컨텍스트 일부를 언로드함
- 다음 애플리케이션이 쓰려 할 때 초기화 비용 발생(1~2초)
- 해결방안 `nvidia-persistenced` daemon
- 애플리케이션이 없어도 드라이버를 로드된 상태로, 하드웨어를 ready 상태로 유지함
- 유휴 시 완전 전원 차단(power gating)을 막음
- 시작 지연과 콜드스타트를 없앨 수 o


3. MPS(Multi-Process Service) 
	- 여러 프로세스가 한 GPU를 공유하면 기본적으로 **time-slicing**. 커널이 짧고 사이에 유휴 갭이 있으면 ping-pong 컨텍스트 스위치만 하다가 GPU가 놀게 됨
	- 한 GPU에 작은 추론 잡을 여러 개 돌릴 때 유용
	- Kubernetes time-slicing — MPS의 대안
		- Device Plugin이 replication factor로 한 GPU에 여러 파드를 시간 분할 배정
		- 실행을 겹치지 않고 기본 드라이버보다 빠르게 전환

- 4. MIG(Multi-Instance Group)
	- hardware level partitioning
	- 가상화지만 하드웨어로 하니 overhead가 몇 % 수준으로 낮음
	- GPU 하나를 최대 **7개**의 논리 GPU로 분할, 각각 전용 메모리와 SM 보유
	- `nvidia.com/mig-2g.45gb:](<http://nvidia.com/mig-2g.45gb:>) "2"``
	- 클러스터 전체에 가용 가능한 자원이 있더라도 그 노드에 없으면 pod가 pending
	- 운영상 걸림돌은 MIG 모드 전환이 GPU 리셋이나 노드 재부팅을 요구 → 잡마다 동적으로 바꿀 수 있는 게 아니라 미리 파티션을 만들어두고 한동안 유지하는 정적 구성
	- GPU Operator의 MIG Manager가 재부팅/드라이버 리로드 후에도 파티션을 유지해주고, 노드에 mig-enabled/mig-disabled 라벨을 붙여 스케줄링을 나누도록 권장


```
nvidia-smi nvlink --status

GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-342a2384-c6d2-809f-f60c-a24ab555a927)
         Link 0: 25 GB/s
         Link 1: 25 GB/s
         Link 2: 25 GB/s
         Link 3: 25 GB/s
         Link 4: 25 GB/s
         Link 5: 25 GB/s
         Link 6: 25 GB/s
         Link 7: 25 GB/s
         Link 8: 25 GB/s
         Link 9: 25 GB/s
         Link 10: 25 GB/s
         Link 11: 25 GB/s
```

5. filesystem overhead
- nfs 사용 ? → serving에서는 nfs 써서 로딩하면 느리다..
- (k8s) model weights 뜰때 node storage에서 바로 물리면 빠른데 storage가 부족함 다른데는 어떻게 해결할까?

6. Networking
- pod 각각은 IP를 갖고 overlay network / NAT를 거치는게 overhead가 될 수 있음  
- 성능이 중요한 GPU 잡에서는 hostNetwork: true(Docker는 --network=host)로 호스트 네트워크를 그대로 쓰는 게 가장 간단한 해법이지만, 보안상 이슈
- NCCL bootstrap  
- NCCL이 NVLink·InfiniBand 등 고속 인터커넥트로 실제 데이터를 주고받기 전, 참여 rank들이 서로를 발견하고 통신 토폴로지(ring/tree)를 합의하는 초기 핸드셰이크 단계
- 고속 경로가 아니라 일반 TCP 소켓으로 수행됨
- 여기서 사용되는 port는 고정값이 아니라 커널이 배정하는 ephemeral 포트임 (리눅스 기본 범위 32768~60999)
- Container 환경에서의 문제
	- host networking(hostNetwork: true) 사용 시에는 호스트 네트워크를 그대로 쓰지만 보안 정책으로 overlay + NetworkPolicy/방화벽을 써야 하는 경우, NCCL이 매 실행마다 다른 포트를 고르기 때문에 사전에 열어줄 포트를 특정할 수 없음
	- 결과적으로 통신이 차단되어 초기화 단계에서 hang이 발생 할 수 있음
- 해결법 : 포트 범위 고정하기
	- NCCL에는 포트 범위를 지정하는 공식 환경변수가 없으므로 kernel sysctl로 해결
	`net.ipv4.ip_local_port_range = 40000 41000`
	- 해당 범위만 NetworkPolicy/방화벽에서 허용