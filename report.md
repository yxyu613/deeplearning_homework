# Stroke2Sketch 细线与发型保持报告

## 1. 问题和版本

内容照片是一位右侧有连续长发的女性，参考图是一位短发儿童铅笔素描。默认结果能保留五官，但头发变成黑块、灰影或断裂线条。本报告的目标是保留照片中的发型外轮廓、发缝、发束方向和耳侧长发，同时让参考图只提供铅笔笔触、明暗和线条密度。

| 版本 | 设置 | 结果 |
| --- | --- | --- |
| [v1](output/archive/v1_initial_no_hair.png) | 默认 TEED、mask、gamma=0.25、SGS=3.5、contrast=1.67 | 头发基本缺失 |
| [v2](output/archive/v2_canny_nomask.png) | 512px Canny、关闭 mask、gamma=0.10、SGS=2.5、contrast=1.0 | 发丝最细腻，但右侧仍有短发块和白色断带 |
| [v3](output/archive/v3_soft_edge.png) | 软 TEED、软 mask、768px 提边后缩回 512px | 头发范围更完整，但整体更灰、更软 |
| [v4](output/archive/v4_plan3.png) | v2 输入路径 + 方案三稀疏 attention + 右发 latent anchor | 右侧发束连续，耳下断带明显减轻 |

四个版本所用的素材已整理到 [output/archive/report_assets/](output/archive/report_assets/)：输入照片为 [原图](output/archive/report_assets/input_photo_original.jpg) / [512×512 实际输入](output/archive/report_assets/input_photo_512.png)，参考素描为 [原图](output/archive/report_assets/reference_sketch_original.jpg) / [512×512 实际输入](output/archive/report_assets/reference_sketch_512.png)。四个版本的对应关系见其中的 [README.md](output/archive/report_assets/README.md)。

版本文件保存在 [output/archive/README.md](output/archive/README.md)。

## 2. 当前推理流程

Stroke2Sketch 使用 Stable Diffusion 1.5、DDPM inversion、跨图像 self-attention、AdaIN 和 FreeU：

    内容照片 → Canny/TEED contour → VAE 4×64×64 latent
      → DDPM inversion → sketch / reference / content / contour 四路 batch
      → UNet self-attention、AdaIN、FreeU → 512×512 素描

四路 batch 的索引为 0=sketch、1=reference、2=content、3=contour。512px 输入经过 VAE 后只有 64×64 latent。发丝如果在预处理阶段消失，skip steps 或 attention 参数不能把它恢复。

## 3. v2 比 v3 细腻的原因

v2 和 v3 使用同一张照片，变化来自 contour：

- v2 在 512px 图像上直接使用 Canny，并关闭 U2Net mask。单根发丝的黑白对比度高，容易形成明确线条。
- v3 使用 TEED 连续概率和 U2Net 软 alpha。弱边缘被保留，但背景也出现中间灰度；发丝和背景的差异在 VAE 压缩后变小。
- v3 在 768px 提边后用 LANCZOS 缩回 512px。它减少第一次缩放丢线，却会把单像素 Canny 发丝的峰值平均掉。
- 旧 attention 仍反复注入参考图。参考图是短发，短发块面会覆盖目标长发的局部线条。

因此 v3 保留了更多低置信度灰度信息，却降低了关键发丝的局部对比度；v2 的锐利 Canny 反而更细腻。当前样例的细线路径是：

    Canny + 不启用 mask + 512px 直接提边
    gamma=0.10, swap_guidance_scale=2.5
    contrast_strength=1.0, skip_steps=30

## 4. 旧方案三为什么没有解决右侧发型

早期方案三尝试连续混合 content/contour 与 reference 的 K/V，并加入 contour token gate，但没有解决右侧发型：

1. 连续 K/V 混合会持续平滑 reference 的高频笔触。
2. K/V token gate 不等于输出空间 gate，会削弱响应，却不能保证右侧发区得到正确结构。
3. content、reference、contour 没有语义配准；即使 K 来自目标照片，V 仍可能携带参考短发。
4. 全图 latent anchor 会把下颌、眼睛等 contour 也拉入 sketch，造成脸部发黑。

实际运行的候选如下：

| 候选 | 结果 |
| --- | --- |
| 调整 Canny、Scharr 和弱边缘阈值 | 右耳下仍是短发断带，并增加噪声 |
| target-K / reference-V | 右下长发仍缺失，纹理变软 |
| attention 输出 token gate | 右上略有改善，但白色断带仍在 |
| 全图 contour latent anchor | 发丝方向改善，但下颌和脸部也被拉黑 |
| 右发 contour latent anchor | 右侧发束连续，耳下断带消失，选用 |

## 5. 最终方案三

### 5.1 稀疏 attention 几何锚定

位置：[run.py:248](run.py:248) 和 [run.py:630](run.py:630)。

上采样 self-attention 只在第 10、20、30、40、50 步注入目标结构：

- 锚定步骤使用完整的 content + contour K/V；
- 其他步骤使用 reference K/V；
- 64×64 和 32×32 token 都遵守该稀疏调度；
- query 继续沿用 v2 的 contour query 混合，gamma=0.10；
- 不使用会衰减高频细节的 K/V 空间 gate。

这样 content/contour 只负责早期发型几何，reference 在其余步骤恢复铅笔线条和明暗。

### 5.2 右发 contour latent anchor

位置：[run.py:456](run.py:456) 和 [run.py:606](run.py:606)。

从 contour 生成 64×64 gate：黑色边缘转成置信度，缩小、轻微高斯平滑、增强弱响应、膨胀连接相邻发丝，再与右侧头发软区域相乘，排除眼睛、下颌和牙齿。

AdaIN callback 后，对 sketch latent 做局部锚定：

    L_sketch ← (1 - lambda*g) * L_sketch + lambda*g * L_contour

参数为：

- callback step 5～42；
- 最大 lambda 0.16，开始和结束处 taper；
- 只修改 latents[0] 的 sketch 分支；
- reference、content、contour 源分支不修改。

照片已有的右侧长发 contour 被轻量拉回 sketch latent，但整张脸不会被 contour 覆盖。当前右发区域使用正面人像的软区域先验；任意姿态泛化时，应换成头发或人物分割区域。

## 6. v4 验证

最终 v4 已覆盖 [output/archive/v4_plan3.png](output/archive/v4_plan3.png)，运行输出为 [output/v4-run/out_ske-skip_steps_30_gamma_0.1-SGS_2.5-CS_1.0.png](output/v4-run/out_ske-skip_steps_30_gamma_0.1-SGS_2.5-CS_1.0.png)。

同一 contour 容差下，右发区域 F1 诊断值为：

| 结果 | F1 |
| --- | ---: |
| v2 | 0.788 |
| 全图 latent anchor | 0.799 |
| 右发 latent anchor | 0.886 |

该 F1 是 contour 一致性代理，不是照片真值。最终照片对照显示：右侧发束从头顶连续到耳下，白色断带减少，孤立短发块和下颌黑块消失，面部铅笔笔触仍接近 v2。

已通过 CUDA 实际推理、python -m py_compile run.py app.py、git diff --check，archive 与运行输出 SHA256 一致。

## 7. 限制和后续

方案三不增加模型和依赖，适合修复 contour 已经包含、但 reference 没有正确表达的发型几何。它不能恢复输入阶段已经丢失的发丝，也不能保证任意姿态都适合当前右发区域先验。

如果动物毛发、草叶、织物或任意姿态人像仍有位置错误，应使用头发/主体分割替换区域先验；若仍不能稳定遵守 contour，再进入方案四的 ControlNet、T2I-Adapter 或显式 line-art residual branch。继续搜索 gamma、swap_guidance_scale 或 skip_steps 不能替代显式边缘条件。
