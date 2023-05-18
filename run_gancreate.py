import multiprocessing as mp
mp.set_start_method('spawn')
# Load model
from IPython.utils import io
import torch
import PIL # 该软件包提供了基本的图像处理功能，如：改变图像大小，旋转图像，图像格式转换，色场空间转换，图像增强，直方图处理，插值和滤波等等
import numpy as np
import ipywidgets as widgets # 用于jupyter笔记本和ipython内核的交互式html小部件
from PIL import Image
import imageio # 提供了一个易于阅读和 编写广泛的图像数据，包括动画图像、体积 数据和科学格式
from models import get_instrumented_model # 在models/wrappers.py下
from decomposition import get_or_compute
from config import Config # 网络的配置
from skimage import img_as_ubyte# 基于python脚本语言开发的数字图片处理包
import hashlib
import flask
# Speed up computation
torch.autograd.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

#@title Load Model for GANSpace
selected_model = 'character' #@param ["portrait", "character", "model", "lookbook"]
latent_dirs = []     # 用于保存潜空间方向的列表
latent_stdevs = []   # 用于保存潜空间标准差的列表

def display_sample_pytorch(seed=None, truncation=0.5, directions=None, distances=None, scale=1, start=0, end=14, w=None, disp=True, save=None):
    # blockPrint()
    model.truncation = truncation
    
    if w is None:
        w = model.sample_latent(1, seed=seed).detach().cpu().numpy() # samplel:样本；即输入网络的图片，在人脸属性编辑是为1，在人脸融合时为2
        w = [w]*model.get_max_latents() # one per layer
    else:
        w = [np.expand_dims(x, 0) for x in w]
    
    if directions != None and distances != None:
        for l in range(start, end): # 从start层到end层
          for i in range(len(directions)):
            w[l] = w[l] + directions[i] * distances[i] * scale
                        #  属性方向        变化距离
            
    torch.cuda.empty_cache() # 删除一些不需要的变量代
    #save image and display
    out = model.sample_np(w)
    
    return out 


def load_model():
    global config     # 全局变量，用于保存模型的配置参数
    global inst       # 全局变量，用于保存经过包装的模型实例
    global model      # 全局变量，用于保存原始的模型实例
    global latent_dirs # 全局变量，用于保存潜空间方向
    config = Config(
        model='StyleGAN2',          # 模型类型，例如StyleGAN2
        layer='style',              # 图像生成层，例如style层
        output_class='character',   # 输出类别，例如character
        components=80,             # 潜空间向量的组件数量
        use_w=True,                 # 是否使用潜空间向量w
        batch_size=5_000,           # 批量大小，用于计算潜空间向量
    )
    inst = get_instrumented_model(config.model, config.output_class,
                                  config.layer, torch.device('cuda'), use_w=config.use_w)  # 获取经过包装的模型实例

    path_to_components = get_or_compute(config, inst) # 获取或计算潜空间向量的路径

    model = inst.model   # 保存原始的模型实例
    comps = np.load(path_to_components)   # 加载潜空间向量文件

    lst = comps.files    # 获取潜空间向量文件中的变量名列表

    latent_dirs = []     # 用于保存潜空间方向的列表
    latent_stdevs = []   # 用于保存潜空间标准差的列表

    load_activations = False   # 是否加载激活值的潜空间方向标志

    for item in lst:
        if load_activations:
            if item == 'act_comp':
                for i in range(comps[item].shape[0]):
                    latent_dirs.append(comps[item][i])    # 提取激活值的潜空间方向
            if item == 'act_stdev':
                for i in range(comps[item].shape[0]):
                    latent_stdevs.append(comps[item][i])  # 提取激活值的潜空间标准差
        else:
            if item == 'lat_comp':
                for i in range(comps[item].shape[0]):
                    latent_dirs.append(comps[item][i])    # 提取原始的潜空间方向
            if item == 'lat_stdev':
                for i in range(comps[item].shape[0]):
                    latent_stdevs.append(comps[item][i])  # 提取原始的潜空间标准差

load_model()

def show_image(params):
    # global latent_dirs # 注释掉的全局变量，潜空间方向
    scale = 1                 # 缩放因子
    seed = params['seed']     # 随机种子，用于生成随机图片
    truncation = params['truncation']   # 截断参数，控制生成图片的多样性

    seed = int(hashlib.sha256(seed.encode('utf-8')).hexdigest(), 16) % 10**8   # 将字符串类型的随机种子转换为整数类型

    start_layer = params['start_layer']     # 开始的生成层
    end_layer = params['end_layer']         # 结束的生成层
    param_indexes = {'gender': 0,
                      'Trueness': 1,
                      'hair': 2,
                      'race': 5,
                      'hairColor': 7,
                      'color': 14,
                      'hairLength': 22}   # 参数与对应的主成分索引的映射关系

    directions = []    # 潜空间方向的列表
    distances = []     # 主成分改变的距离的列表
    p = ['seed','truncation', 'start_layer', 'end_layer']   # 不参与主成分计算的参数列表

    # 提取主成分方向和对应的距离
    for k, v in params.items():
        if k not in p:
            directions.append(latent_dirs[param_indexes[k]])  # 对应主成分的方向
            distances.append(v) # 对应主成分改变的距离

    style = {'description_width': 'initial'}   # 控制显示样式的字典
    # 调用display_sample_pytorch函数生成图片，并返回生成的图片
    return display_sample_pytorch(int(seed), truncation, directions, distances, scale, start = int(start_layer), end = int(end_layer), disp=False)


app = flask.Flask(__name__)
# 开始服务
@app.route("/predict", methods=["POST"])
def predict():
    if flask.request.method == "POST": # 通过POST发送请求 

        if flask.request.files.get("image"): # 接受image数据
            data = flask.request.files["image"].read()
            data = eval(data)
            result = show_image(data) # nparray


    return flask.jsonify(result.tolist())


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    print("start")
    # load_model()

    app.run(host="0.0.0.0", port=8000)
    



    # params = {
    #   'seed':'112',
    #   'truncation': '0.5',

    #   'monster': '0.5',
    #   'female': '0',
    #   'skimpy': '10',
    #   'light': '10',
    #   'bodysuit': '0',
    #   'bulky': '0',
    #   'human_head': '0',

    #   'start_layer': '0',
    #   'end_layer': '0'
    #   }
    # out = show_image(params)
    # final_im = Image.fromarray((out * 255).astype(np.uint8)).resize((500,500),Image.LANCZOS) # 实现array到image的转换
    # # print(final_im)
    # final_im.save(f'im1.png')
