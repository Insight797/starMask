# StarMask — 银河摄影缩星工具

把一张银河照片拆成：

| 输出 | 内容 |
|------|------|
| `*_stars.tif` | **纯星点**，背景为黑 |
| `*_starless.tif` | **银河 + 星云 + 地景**，去掉星点 |
| `*_starmask.tif` | 星点软掩膜（可选） |

数学上两张图互补：`stars + starless ≈ 原图`，可直接进 Photoshop 缩星：

1. 叠加 *stars* 图层  
2. 用色阶/曲线/最小值滤镜缩小或压暗星点  
3. 与 *starless* 合成 → 银河更干净、更突出  

## 支持的格式

**RAW（各厂通用，走 LibRaw）**  
Nikon `.nef` `.nrw` · Sony `.arw` · Canon `.cr2` `.cr3` · Fujifilm `.raf` · Olympus `.orf` · Panasonic `.rw2` · Pentax `.pef` · Sigma `.x3f` · Adobe `.dng` 等  

**常规图像**  
16-bit `.tif` / `.tiff` / `.exr` · `.jpg` / `.png`

## 安装

```bash
cd starMask
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

依赖：`rawpy`（LibRaw）、`numpy`、`scipy`、`tifffile`、`click`。

## 图形界面（最快上手）

```bash
./run_gui.sh
# 或
.venv/bin/starmask gui
```

深色摄影风：添加 RAW/文件夹 → 选中预览（原图｜无星｜星点）→ 开始缩星 → 导出互补 TIFF。

## 使用

```bash
# 处理单张 RAW（尼康 NEF / 索尼 ARW 均可）
starmask process DSC_1234.NEF --method starnet

# 指定输出目录，多文件 / 整个文件夹
starmask process photo.ARW -o ./out
starmask process ./raw_folder/ -o ./out

# 调参
starmask process sky.CR3 --method starnet

# 无 StarNet2 时回退经典 PSF
starmask process sky.CR3 --method psf --threshold 4.5

# 先看图上的星点尺度
starmask info milkyway.NEF
```

### 参数说明

| 参数 | 默认 | 说明 |
|------|------|------|
| `--method` | `starnet` | `psf`：检测星点并拟合高斯 PSF 扣除（最准，保银河）  
| | | `mask`：软掩膜 + 局部背景替换（更快）  
| | | `morph`：形态学 top-hat（最快，适合预览） |
| `--fwhm` | 自动 | 星点半高全宽（像素）。长焦/短焦差异大时可手动指定 |
| `--threshold` | `4.5` | 星点检测阈值（噪声倍数）。银河很亮时可略提高 |
| `--gain` | `0.95` | 扣除增益；`<1` 留一点星晕，`1.0` 更干净 |
| `--max-stars` | `8000` | 最多处理星数（银心稠密区） |
| `--preview` | off | RAW 优先读内嵌预览，批量预览更快 |
| `--no-mask` | off | 不导出 mask |
| `--compare` | off | 额外输出 原图\|无星\|星点 三联对比 PNG |

## 算法说明

| 方法 | 原理 | 质量 |
|------|------|------|
| **`starnet`（推荐）** | **StarNet2 卷积神经网络**，端到端预测无星图（与市面成熟去星工具同源引擎） | 银河图可近乎零星点 |
| `psf` | 星点检测 + 椭圆高斯 PSF 拟合扣除 + 残差补净 | 缩星可用，密集星野有残留 |
| `mask` / `morph` | 软掩膜替换 / 形态学 top-hat | 快速预览 |

### 安装 StarNet2 引擎（强烈推荐）

```bash
bash scripts/install_starnet2.sh
# 或从 https://starnetastro.com/cli-tools/starnet/ 手动下载 CLI
# 然后: export STARNET2_BIN=/path/to/starnet2
```

处理：

```bash
starmask process 照片.NEF --method starnet -o ./out --compare
```

StarNet2 会输出无星图；星点层 = 原图 − 无星图，保证互补。

> 神经网络（StarNet / StarNet2 / StarXTerminator）才能做到「银河图完全无星」。
> 传统检测+扣星只能做到「缩星」，密集银心必然残留——这是方法上限，不是调参问题。

## Python API

```python
from starmask import separate_image
from starmask.rawio import load_image, save_tiff

img = load_image("milkway.ARW")
result = separate_image(img, method="starnet")
save_tiff("out_stars.tif", result.stars)
save_tiff("out_starless.tif", result.starless)
print(len(result.star_field.stars), "stars")
```

## 测试

```bash
pytest -q
```

会用合成银河图（带星点 + 尘埃带 + 地景）验证检测数量、互补性、背景干净度。

## 后续建议

- 高感短曝光：可把 `--threshold` 提到 6–7，去掉热噪。  
- 星野密集（银心）：可增大 `--max-stars`，或先 `starmask info` 看检测量。  
- 地景占比较大时，算法会自动少在地面扣星（星点检测不过阈值）。

## 输出格式（TIF / DNG）

默认导出 16-bit TIFF。若要做 **AI 降噪**（Lightroom / Camera Raw 等），请改用 **线性 DNG**：

```bash
starmask process photo.ARW --method starnet --reduce --format dng
starmask process photo.ARW --method starnet --reduce --format both   # tif + dng
```

| 格式 | 用途 |
|------|------|
| `tif` | 通用，PS / 像素编辑 |
| `dng` | **Linear DNG**（16-bit LinearRaw），可进 LR/ACR 的 AI Denoise |
| `both` | 两套都写 |

GUI 右侧「输出格式」可选 `dng` / `tif` / `both`（默认 dng）。

> 说明：这是**线性 DNG**（已去马赛克的 RGB），不是相机原始 Bayer DNG。  
> Lightroom Classic / Camera Raw 的 AI 降噪支持 Linear DNG；Topaz 等也可直接吃 TIFF。  
> 想让 DxO PureRAW 这类「只吃原始 RAW」的软件降噪，请对**原片**先降噪再进 StarMask。

## 缩星（保留亮星）

不把星全抹掉，只去掉暗弱/低质量星点，高亮星自动保留作视觉锚点：

```bash
starmask process photo.ARW --method starnet --reduce
# 手动指定保留最亮的 10%
starmask process photo.ARW --method starnet --reduce --keep-ratio 0.10
```

| 输出 | 含义 |
|------|------|
| `*_reduced.tif` | **最终缩星图**（银河+地景+保留的亮星） |
| `*_stars_kept.tif` | 保留下来的亮星 |
| `*_stars_removed.tif` | 去掉的暗星 |

关系：`stars_removed + reduced ≈ 原图`，`stars_kept + stars_removed ≈ stars`。

自动分层：优先在亮度分布上找「暗星海 / 高亮星」双簇（Otsu），连续分布则取最亮约 10%。GUI 里勾选「缩星」并可拖动保留比例。

## 质量验收

```bash
.venv/bin/pytest -q
.venv/bin/python scripts/auto_qa.py
```

硬指标：
- `stars + starless ≈ 原图`（互补误差 &lt; 1e-3）
- 银河图内部残留点源 ≤ 12（StarNet2 通常 0–6）
- 星点图地景结构泄漏 ≈ 0（地景上的真星仍保留在星点图）
- 星点图背景 p99 ≤ 0.02

## 备注

- 尼康「高效压缩」NEF（如 OJPEG 封装）LibRaw 无法全量 demosaic，会自动回退内嵌预览；需要全分辨率请在相机改为「无损压缩 RAW」，或先导出 16-bit TIFF 再处理。
- StarNet2 权重由 CLI 在真实二进制旁自动查找，无需手动指定 `-w`。
