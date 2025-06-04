
# RGB-Thermal Translation using Super Resolution

This project explores RGB-Thermal (RGB-T) image translation using a combination of deep learning and super-resolution techniques. It aims to improve translation quality between RGB and thermal images by integrating a super-resolution module into the GAN training pipeline.

## 🔍 Overview

The pipeline consists of:
- A GAN-based image translation model.
- A super-resolution module (SR).
- Data loading and preprocessing utilities.
- Training, validation, and testing scripts.

It is designed to investigate the effect of super-resolution on RGB-T image translation, particularly focusing on enhancing thermal image quality.

## 📂 Project Structure

```
├── create_sr_images.py     # Generate SR images for training
├── dataset.py              # Dataset preprocessing
├── models/                 # Model architecture definitions
├── test.py                 # Testing script
├── train_val.py            # Training and validation script
├── utils/                  # Utility functions
├── requirements.txt        # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- CUDA-enabled GPU or Apple MPS backend (optional but recommended)

### Installation

Clone the repository:

```bash
git clone https://github.com/alperendemirciai/RGB-Thermal-Translation-using-Super-Resolution.git
cd RGB-Thermal-Translation-using-Super-Resolution
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Download the model weights from the link below:

https://drive.google.com/drive/folders/1chQCeUN_lguLqcPKeEb7IeY0cYt0WYS_?usp=share_link 



### Dataset

Prepare your dataset in the following structure:

```
labeled_rgbt_pairs/
├── color/
├── sr_3x/
├── sr_2x/
├── thermal8/
```

Ensure that filenames match for paired RGB and thermal images.

## 🏋️‍♀️ Training

To train the model:

```bash
python train_val.py --data ./labeled_rgbt_pairs --save_dir ./outputs/13th_sr3x_run --lr 0.0005 --batch_size 32 --epochs 100 --print_every 2 --upsampling_method pixelshuffle --gen_filters 32 --disc_filters 32 --random_state 42 --thermal_type thermal8 --sr sr_3x --save_freq 3
```

## 📊 Testing

To evaluate the trained model:

```bash
python test.py --data ./labeled_rgbt_pairs --save_dir ./outputs/13th_og_run --batch_size 32 --upsampling_method pixelshuffle --gen_filters 32 --random_state 42 --thermal_type thermal8 --sr color
```

## 🧪 Generate Super-Resolved Images

```bash
python create_sr_images.py   --input_dir ./input_images   --output_dir ./sr_output_images
```

## 📈 Results

You can visualize training outputs and generated images from the `outputs/` directory. Include examples in the repo if applicable.

## 🤝 Contributing

Feel free to fork the repo and submit pull requests. Bug reports and feature suggestions are welcome!

## 📄 License

This project is open source under the MIT License. See the [LICENSE](LICENSE) file for details.
