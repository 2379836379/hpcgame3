### hpcgame-3rd writeup

> 使用模型：gpt-5.2 / gpt-5.2-codex
#### A.签到
搜索得知，代码是 Fortran 语言一个输出自身源代码的程序。复制代码提交完成题目

#### B.小北问答
使用网页版chatgpt即可完成绝大多数题目，唯一不能解决的题目是

7. GPU

    NVIDIA 的 Hopper 架构引入了 TMA（Tensor Memory Accelerator） 以提升 GPU 内存访问效率。以下说法正确的有：

    A :	相比 cp.async，TMA 可以直接将数据从全局内存加载到共享内存，无需经过寄存器中转，从而能节省寄存器

    B :	在 cutlass 的异步流水线抽象中，Producer 调用 producer_acquire 获取空闲的 buffer stage，完成数据加载后调用 producer_commit 通知 Consumer；Consumer 则通过 consumer_wait 等待数据就绪，使用完毕后调用 consumer_release 释放 buffer

    C :	在使用 TMA 进行数据传输时，所有参与的线程都需要执行相同的 TMA 指令，TMA 硬件会自动处理线程间的协调

    D :	Cutlass Pipeline 使用多级缓冲（multi-stage buffering），通过 PipelineState 追踪当前读写的 stage index 和 phase，实现 Producer 和 Consumer 之间的流水线重叠

    E :	TMA 的 multicast 功能允许一次 TMA 操作将同一块数据广播到 Cluster 内的多个 Thread Block 的共享内存中，减少了重复的全局内存访问

    F :	TMA 描述符（TMA Descriptor）需要在 kernel 启动前在 host 端创建，描述符中包含了张量的形状、步长和 swizzle 模式等信息，kernel 执行时通过预取描述符（prefetch_tma_descriptor）来减少首次 TMA 操作的延迟

换用codex cli后立即得到正确答案

#### C.Ticker
修改struct：
```
struct Candle {
    double high;
    double low;
    double close;
    long long vol;
};
```
由于strut未对齐cacheline导致无法有效并行，尝试对齐到64byte发现无法满分。查询鲲鹏 CPU 的硬件特质后对齐到128byte得到满分。

#### D.Hyperlane Hopper
题目一个经典的单源最短路 (Single Source Shortest Path, SSSP) 问题，绝大部分由ai完成
```
Algorithm : Delta-Stepping 

1.Bucket 思想：
把距离按区间 [k*delta, (k+1)*delta) 分桶：
桶内用轻边（w <= delta）扩展，逐步清空该桶
2.关键数据结构：
bucket_head[b]：桶头链表索引
pool[]：节点池，避免频繁分配
每线程局部 push list，再批量 splice 到全局桶
3.并行性：
relax 过程采用 __atomic_compare_exchange_n
每线程局部缓冲，批量合并到桶，减少锁争用
4.Correctness Sketch：
图边权非负 → Delta-Stepping 保证正确性
每次 relax 只在 nd < old 时更新，原子 CAS 确保并发安全
桶内闭包确保轻边收敛后再处理重边，等价于 Dijkstra 分层扩展
```
调试过程中发现距离满分差距较大，尝试过程中发现或可对小规模情况增加特判：
```
特定小规模用例（n=1e5, m=2e5） 走 calculate1：并行 label-correcting（worklist）算法；
其余情况 走并行 Delta-Stepping 算法，配合 CSR 图存储与桶化队列加速。
整体设计目标是：在不同规模和密度的图上保持稳定的并行性能。

Algorithm : Sparse Case
用于特定小规模测试点（n==1e5 && m==2e5）：
类似 label-correcting 的并行 worklist
维护 frontier（当前队列），每轮并行 relax 相邻边
每线程写入自己的 local_next，最后合并成下一轮 frontier
使用 compare_exchange 保证距离更新正确
```
得到程序距离满分差距仍较大，且无明显调试改进，最终未能得到满分答案。尝试调整delta值提升速度，最终最佳一次尝试的得分为 80.42/100

#### E.哪里爆了
由于初始范围过大，未使用ai进行完整分析，人工尝试后由于未能定位问题未能得分。

#### F.小北买文具
矩阵LU分解问题，绝大部分使用ai完成
```
1. 流程
对 A 做分块 LU 分解（面板分解 + 尾随更新），记录主元行 ipiv；
将行置换应用到右端向量 b；
前代解 L y = b；
回代解 U x = y，并把解写回 b。
2. 分块策略与面板分解
采用固定块大小 BS，通过 for (k=0; k<n; k+=BS) 逐块处理。
面板分解对 k..k+BS 的列进行 部分选主元，计算该面板的 L 和 U。
当前实现为：
先扫描该列最大值（行优先下为跨步访问），找到 pivot；
若 |A(j,j)| 明显小于该列最大值，则进行行交换；
更新面板内部的上三角部分（保持计算局部性）。
3. 行交换策略（面板内 + 尾随批量）
面板阶段只交换 左侧 + 面板区域 [0, k+kb)，避免在面板里频繁扫大矩阵；
面板结束后，再批量把交换应用到尾随矩阵 [k2, n)。
这样做减少了对大矩阵的无意义搬运，改善缓存友好性。
4. 并行策略（单并行区）
使用 单个 #pragma omp parallel 并行区，避免每个 block 反复 fork/join。
面板分解与 U12 计算由 #pragma omp single 保证单线程执行（保持数值一致性和简化同步）。
尾随矩阵更新用 #pragma omp for schedule(static) 并行化，这是主要的计算热点。
5. 阈值选主元（减少交换次数）
采用阈值选主元：只有当对角元明显小于该列最大值时才交换：
do_swap = (|A(j,j)| < PIV_THRESH * max_col)
这样可以显著减少行交换次数（降低内存搬运成本），同时在数值稳定性允许的范围内保持精度。
```
初始未得到满分，过程中发现对n不同的矩阵调整分块大小能够提升性能，经过一定尝试后得到分界：
```
// 分块大小
    int BS = 48;
    if(n > 16384) BS = 64;
    if(n == 8192) BS = 32;
```
尝试后仍未满分，由于提交次数不足无法继续尝试，最终得分 97.82/100

#### G.显存不足
未得分
#### H.流水排排乐
大部分内容可由ai完成，仅第三、四部分需要手动修改
```
任务 1：向量加 (Task 1)
并行策略：固定 1 个 job 处理整向量。
SPM 策略：双缓冲 + 就地输出，块大小选择使 SPM 满载但不超限。
原因：单 SM + 高 Lmem 下，减少 job 切分带来的调度开销；双缓冲用于覆盖 load latency。
```
```
任务 2：矩阵乘法 (Task 2)
分块策略：TILE_M=128, TILE_N=128, TILE_K=128。
job 数：(M/128)*(N/128)，尽量拉满 32 SM。
核心优化：A/B 双缓冲，C 在 SPM 内累加，K 维分块循环。
原因：MXM 单元吞吐高，pad 规则要求 M/N 到 128、K 到 16 的倍数，tile 选择直接决定效率。
```
```
任务 3：带细粒度缩放的矩阵乘法
问题特点：A/B 分别由 base 与 scale 组成，K 方向每 64 元素共享一个 scale。核心开销不是单纯的 MXM，还要在每个 K-block 前把 Ab/Bb 进行缩放。
总体策略：沿 K 方向分块，先把每个块的 Ab/Bb 用对应 scale 乘起来，再做 matmul 累加到 C。为了降低 Lmem 访问代价，把 As/Bs 对应的整块一次性加载到 SPM。
```
测试发现无法得到满分，对两组测试点进行特判，进行不同的分块配置后得到满分

    M=512 ：TILE_M=128, TILE_N=128, TILE_K=128
    M=2048 ：TILE_M=256, TILE_N=128, TILE_K=128
```
任务 4：Flash Attention
整体思路：按 FlashAttention v1 的块流式 softmax 思路实现，避免显式构建完整注意力矩阵。每个 job 处理一个 Q 的 block，遍历所有 K/V block，边算边更新 O。
分块策略：设定 BLOCK_Q = 128、BLOCK_K = 128、D = 128。num_jobs = S_qo / BLOCK_Q
流水与缓存：使用双缓冲 k0/v0、k1/v1 与 score0/score1，在计算当前块时预取下一个 K/V，尽量减少 Lmem 访问阻塞。
收尾：循环结束后做 O = O / l（按行广播），并写回输出。
```
测试发现同样不能得到满分，尝试更改流水线至三级，但由于空间限制导致每级流水线缓存大小减小，综合速度降低，后改回二级流水线。继续调试发现每个周期内指令的执行顺序未进行调整，占用MXM且速度最慢的matmul指令其中一个处于周期末尾，将不存在数据依赖的matmul调整至每周期开始阶段，测试得到满分

#### I.Python 笑传之吃吃饼
本题全部内容由ai完成
```

1. 目标与核心思路
题目要求从拉格朗日函数自动生成加速度的计算程序。

2. 计算图与符号追踪
实现了一个轻量的计算图系统：
_Node：记录操作类型及子节点
_TraceScalar：重载 + - * / sin cos pow，将 Python 运算映射成计算图
_TraceArray：为向量生成可索引的 TraceScalar
_NODE_CACHE：对相同子表达式进行节点复用（CSE）
这样用户传入的可以被自动构建为表达式树。

3. 符号微分与结构检测
通过递归规则计算一阶/二阶导数：
_diff：对计算图做符号微分（支持 add/sub/mul/neg/sin/cos）
自动得到 f_q, f_v, f_qv, f_vv
进一步识别特殊结构：
对角形式（C/M 仅对角）：直接用逐元素公式计算 
常量质量矩阵：预计算 M 并做 Cholesky / 逆矩阵优化

4. 代码生成与 C 编译
对符号表达式做拓扑排序，生成 C 代码：
中间变量 t0, t1, ... 线性展开，避免重复计算
三种生成策略：
diag：对角情况
full：一般线性系统，Gaussian elimination + 回代
full_unrolled：n=20 时完全展开（减少循环开销）
full_const_m：常量 M 时，Cholesky 或逆矩阵直接求解
gcc 编译为 .so，并通过 ctypes 动态加载
编译缓存使用代码内容 + flags 的 hash，避免重复编译。

5. 运行时分级回退
构建失败时，按优先级降级
```
经过反复调试成绩均较低，疑似存在整体方法上的问题，经过多次细微调整最终得分 50.98/100

#### J.古法 Agent
本题使用ai即可获得满分
```
算法设计
1) 双哈希 + 预生成“1 位差”变体
对每个模式串 P：
计算长度 64 的双 64-bit 哈希 (h1, h2)，并插入哈希表计数。
对每一位置 i，将该位替换为 26 个字母中除原字母的 25 个候选，得到所有“一位差”变体的哈希 (h1', h2')，同样插入哈希表计数。
这样哈希表中记录了：
完全匹配的哈希
所有允许 1 位差的哈希
当文本窗口哈希命中时，直接累加该哈希对应的计数即可得到匹配总数。
2) 文本滚动哈希扫描
对文本 T 进行长度 64 的滑窗：
计算首个窗口哈希 (h1, h2)
使用滚动哈希 O(1) 更新下一个窗口
查表累加匹配数
3) OpenMP 并行
将 N-64+1 个窗口均匀分配给线程：
每个线程独立维护哈希状态
使用 OpenMP reduction(+:total_matches) 汇总结果
4) 功耗控制与工程细节
通过后台线程周期性调用功耗接口，得到 CPU + 其他部件 总功耗。
若超过限制，设置 throttle 标志，工作线程每隔固定窗口数短暂 sleep（200µs），实现轻量节流。
使用 baseline 中的 cpuset 解析与绑核逻辑，避免环境绑核异常影响性能。
采用 mmap 读取输入文件，减少 I/O 开销。
```
调试生成的答案后在本地调试通过generator验证正确性，提交即获得满分

#### L.稀疏注意力
未得分

#### M.真忙的管理员-MapReduce SpGEMM
本体内容全部由ai完成
```
本题要求用 MPI 的 MapReduce 思路做稀疏矩阵乘法并输出每行 top-K。
实现按照题面“两轮 shuffle”的设计完成：
总体流程
1) 以 k 为键做第一次 shuffle  
   - A 端按 k=c 分发，B 端按 k=r 分发  
   - 同一 k 的 A(i,k) 与 B(k,j) 会落到同一 rank  
2) 本地做外积累加  
   - 先将 B 按 k 分组：B_map[k] = [(j, val)]  
   - 遍历 A_by_k：对每个 A(i,k,val) 与 B_map[k] 相乘  
   - 用 (i,j) 作为 key 做累加  
3) 以 i 为键做第二次 shuffle  
   - 把 (i,j,partial) 按 i 分发到同一 rank  
   - 本地汇总得到最终 (i,j)  
4) 每行 top-K  
   - 对每个 i 的 (j,score) 做 nth_element + sort  
   - 输出 i 及其 topK 结果  

关键实现细节
- Shuffle：Alltoall 交换计数，Alltoallv 发送 Triplet  
- Key 设计：用 uint64 拼 (i<<32 | j) 作为哈希 key，降低嵌套 map 开销  
- 累加精度：double  
- top-K：若 size>K 先 nth_element 截断，再全排序  

总体效果是把完整 SpGEMM 拆成“按 k 聚合的局部外积 + 按 i 归并”的两阶段 MapReduce，
避免某一行被拆散在多个 rank 上，符合题目的分布式处理要求。
```
由于本地设备运行MPI存在未知故障且最终未能修复，未进行本地测试，最终提交得分 29.09/100