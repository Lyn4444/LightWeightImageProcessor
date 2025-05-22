import torch
from torch import nn
from torch.utils.mobile_optimizer import optimize_for_mobile

from app.src.main.assets.models_M2.fpn_micronet_large import FPNMicroNet_Large

# 1. 加载您的预训练模型
#    确保您知道模型的结构以便正确加载。
#    如果您的 .pt 文件仅包含模型权重 (state_dict)，您需要先实例化模型类，然后加载权重。
#    如果您的 .pt 文件是使用 torch.save(model, PATH) 保存的整个模型，则可以直接加载。

# 示例：如果您的 .pt 是 state_dict
# from your_model_definition_file import YourModelClass # 导入您的模型类定义
# model = YourModelClass()
# model.load_state_dict(torch.load('assets/deblur_large.pt'))



model = FPNMicroNet_Large(nn.BatchNorm2d)

try:
    model.load_state_dict(torch.load('/home/lyn/Downloads/LightWeightImageProcessor/app/src/main/assets/deblur_large.h5'))
except Exception as e:
    print(f"尝试直接加载模型失败: {e}")
    print("如果 '/home/lyn/Downloads/LightWeightImageProcessor/app/src/main/assets/deblur_large.h5' 只包含权重 (state_dict),")
    print("请确保先实例化模型类，然后使用 model.load_state_dict() 加载权重。")
    exit()

# 2. 将模型设置为评估模式
#    这对于像 Dropout 和 BatchNorm 这样的层很重要。
model.eval()

# 3. 创建一个示例输入张量
#    这个张量的形状和类型应该与您的模型期望的输入一致。
#    例如，如果您的模型处理 3 通道、224x224 像素的图像：
#    example_input = torch.rand(1, 3, 224, 224)
#    您需要根据 'deblur_large.pt' 模型的实际输入调整此处的尺寸。
#    假设您的去模糊模型输入是单通道 256x256 的图像：
try:
    # 尝试获取模型输入的形状信息（如果模型有此属性）
    # 这只是一个猜测，实际情况取决于模型是如何定义的
    # 如果没有，您需要手动指定
    if hasattr(model, 'input_shape'):
        batch_size, channels, height, width = model.input_shape
        example_input = torch.rand(batch_size, channels, height, width)
    else:
        # 请根据您的 'deblur_large' 模型的实际输入形状进行修改
        print("警告: 无法自动确定模型输入形状。使用默认形状 (1, 1, 256, 256)。")
        print("请确保这与您的模型期望的输入匹配。")
        example_input = torch.rand(1, 1, 256, 256) # 示例：(batch_size, channels, height, width)
except Exception as e:
    print(f"创建示例输入时出错: {e}")
    print("请手动创建一个与模型输入匹配的 'example_input'。")
    exit()


# 4. 跟踪模型以生成 TorchScript 模块
#    `torch.jit.trace` 会运行一次模型，记录下输入张量在模型中的操作流程。
try:
    traced_script_module = torch.jit.trace(model, example_input)
except Exception as e:
    print(f"模型跟踪失败: {e}")
    print("确保 'example_input' 的形状和类型与模型期望的一致。")
    exit()

# 5. (可选但推荐) 优化 TorchScript 模块以用于移动部署
try:
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
except Exception as e:
    print(f"移动端优化失败: {e}")
    # 如果优化失败，您仍然可以尝试使用未优化的 traced_script_module
    traced_script_module_optimized = traced_script_module


# 6. 保存优化后的 TorchScript 模块 (通常为 .ptl 文件)
#    这个文件将包含在您的 Android 应用的 assets 目录中。
#    目标路径可以根据您的 Android 项目结构进行调整。
output_path = "/home/lyn/Downloads/LightWeightImageProcessor/app/src/main/assets/deblur_large_mobile.ptl" # 示例路径
try:
    traced_script_module_optimized._save_for_lite_interpreter(output_path)
    print(f"模型已成功转换为 TorchScript 并保存到: {output_path}")
except Exception as e:
    print(f"保存优化模型失败: {e}")
    # 备用方案：如果 _save_for_lite_interpreter 失败，可以尝试普通的 save
    try:
        alternative_output_path = "/home/lyn/Downloads/LightWeightImageProcessor/app/src/main/assets/deblur_large_mobile_unoptimized.pt"
        traced_script_module.save(alternative_output_path)
        print(f"警告: Lite interpreter 保存失败。模型已保存到 (未优化): {alternative_output_path}")
    except Exception as e_save:
        print(f"保存模型时发生严重错误: {e_save}")