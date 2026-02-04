import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==========================================
# 1. 設定 (ここだけ確認してください)
# ==========================================
BASE_DIR = r'C:/Users/TakedaYuya/Landmark_Gaihouzu_new'
INPUT_IMG_PATH = os.path.join(BASE_DIR, 'input/NI_52_11_8.jpg')  # 入力画像
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')                     # 出力先
# ファイルの場所を直接指定します
CHECKPOINT_PATH = r'C:\Users\TakedaYuya\Landmark_Gaihouzu_new\DexiNed-master\checkpoints\10_model.pth'

# ==========================================
# 2. DexiNed モデル定義 (model.pyの中身を移植)
# ==========================================
def weight_init(m):
    if isinstance(m, (nn.Conv2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, mean=0.0)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    if isinstance(m, (nn.ConvTranspose2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, mean=0.0)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class CoFusion(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CoFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, out_ch, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.norm_layer1 = nn.GroupNorm(4, 64)
        self.norm_layer2 = nn.GroupNorm(4, 64)
    def forward(self, x):
        attn = self.relu(self.norm_layer1(self.conv1(x)))
        attn = self.relu(self.norm_layer2(self.conv2(attn)))
        attn = F.softmax(self.conv3(attn), dim=1)
        return ((x * attn).sum(1)).unsqueeze(1)

class _DenseLayer(nn.Sequential):
    def __init__(self, input_features, out_features):
        super(_DenseLayer, self).__init__()
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(input_features, out_features, kernel_size=3, stride=1, padding=1, bias=True)),
        self.add_module('norm1', nn.GroupNorm(4, out_features))
    def forward(self, x):
        x1, x2 = x
        new_features = super(_DenseLayer, self).forward(F.relu(x1))
        return 0.5 * (new_features + x2), x2

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, input_features, out_features):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_features, out_features)
            self.add_module('denselayer%d' % (i + 1), layer)
    def forward(self, x):
        x_out, x_in = x
        for name, layer in self.named_children():
            x_out, x_in = layer((x_out, x_in))
        return x_out

class UpConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.shuff = nn.PixelShuffle(2)
    def forward(self, x):
        x = self.conv(x)
        x = self.shuff(x)
        return x

class SingleConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SingleConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.norm = nn.GroupNorm(4, out_ch)
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x

class DoubleConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.norm = nn.GroupNorm(4, out_ch)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.norm(x)
        return x

class DexiNed(nn.Module):
    def __init__(self):
        super(DexiNed, self).__init__()
        self.block_1 = DoubleConvBlock(3, 32)
        self.block_2 = DoubleConvBlock(32, 64)
        self.dblock_3 = _DenseBlock(2, 64, 128)
        self.dblock_4 = _DenseBlock(3, 128, 256)
        self.dblock_5 = _DenseBlock(3, 256, 512)
        self.dblock_6 = _DenseBlock(3, 512, 512)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.side_1 = SingleConvBlock(32, 128)
        self.side_2 = SingleConvBlock(64, 128)
        self.side_3 = SingleConvBlock(128, 128)
        self.side_4 = SingleConvBlock(256, 128)
        self.side_5 = SingleConvBlock(512, 128)
        self.side_6 = SingleConvBlock(512, 128)

        self.pre_dense_2 = SingleConvBlock(128, 256)
        self.pre_dense_3 = SingleConvBlock(128, 256)
        self.pre_dense_4 = SingleConvBlock(128, 256)
        self.pre_dense_5 = SingleConvBlock(128, 256)
        self.pre_dense_6 = SingleConvBlock(128, 256)

        self.up_block_1 = UpConvBlock(128, 1)
        self.up_block_2 = UpConvBlock(128, 1)
        self.up_block_3 = UpConvBlock(128, 1)
        self.up_block_4 = UpConvBlock(128, 1)
        self.up_block_5 = UpConvBlock(128, 1)
        self.up_block_6 = UpConvBlock(128, 1)

        self.block_cat = CoFusion(6, 6)
        self.apply(weight_init)

    def forward(self, x):
        # Encoder
        x1 = self.block_1(x)
        x2 = self.block_2(self.maxpool(x1))
        x3 = self.dblock_3([self.maxpool(x2), self.maxpool(x2)])
        x4 = self.dblock_4([self.maxpool(x3), self.maxpool(x3)])
        x5 = self.dblock_5([self.maxpool(x4), self.maxpool(x4)])
        x6 = self.dblock_6([self.maxpool(x5), self.maxpool(x5)])

        # Side Outputs
        s1 = self.side_1(x1)
        s2 = self.side_2(x2)
        s3 = self.side_3(x3)
        s4 = self.side_4(x4)
        s5 = self.side_5(x5)
        s6 = self.side_6(x6)

        # Upsampling blocks
        o1 = self.up_block_1(s1)
        o2 = self.up_block_2(s2 + self.pre_dense_2(s1))
        o3 = self.up_block_3(s3 + self.pre_dense_3(s2))
        o4 = self.up_block_4(s4 + self.pre_dense_4(s3))
        o5 = self.up_block_5(s5 + self.pre_dense_5(s4))
        o6 = self.up_block_6(s6 + self.pre_dense_6(s5))
        
        # 融合
        o6 = F.interpolate(o6, size=x.shape[2:], mode='bilinear', align_corners=False)
        o5 = F.interpolate(o5, size=x.shape[2:], mode='bilinear', align_corners=False)
        o4 = F.interpolate(o4, size=x.shape[2:], mode='bilinear', align_corners=False)
        o3 = F.interpolate(o3, size=x.shape[2:], mode='bilinear', align_corners=False)
        o2 = F.interpolate(o2, size=x.shape[2:], mode='bilinear', align_corners=False)
        o1 = F.interpolate(o1, size=x.shape[2:], mode='bilinear', align_corners=False)

        fuse = self.block_cat(torch.cat((o1, o2, o3, o4, o5, o6), dim=1))
        return [o1, o2, o3, o4, o5, o6, fuse]

# ==========================================
# 3. 実行処理
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. 画像ロード
    if not os.path.exists(INPUT_IMG_PATH):
        print(f"Error: 画像がありません {INPUT_IMG_PATH}")
        return
    
    img_org = cv2.imread(INPUT_IMG_PATH)
    h, w = img_org.shape[:2]
    print(f"Original Size: {w}x{h}")

    # 2. リサイズ（GPUメモリ対策: 大きすぎると落ちるため）
    # 精度を上げるならここを大きくする (例: 1024, 1024)
    process_size = (1024, 1024) 
    img_resized = cv2.resize(img_org, process_size)
    
    # テンソル化
    img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0)
    # 平均値を引く (DexiNedの仕様)
    img_tensor -= torch.tensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1)
    img_tensor = img_tensor.to(device)

    # 3. モデルロード
    model = DexiNed().to(device)
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: 重みファイルがありません {CHECKPOINT_PATH}")
        return
        
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print("Model Loaded.")

    # 4. 推論
    with torch.no_grad():
        preds = model(img_tensor)
        # 最後の出力が融合エッジ
        pred_fuse = preds[-1]
        
    # 5. 画像化して保存
    pred_fuse = torch.sigmoid(pred_fuse).cpu().numpy()[0, 0]
    # 元のサイズに戻す
    pred_fuse = cv2.resize(pred_fuse, (w, h))
    # 0-255に
    result = (pred_fuse * 255).astype(np.uint8)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, 'dexined_edge.png')
    cv2.imwrite(out_path, result)
    print(f"Save: {out_path}")

if __name__ == '__main__':
    main()