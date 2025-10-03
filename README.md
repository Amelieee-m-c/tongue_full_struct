# Tongue-AI-V2

## 📂 專案腳本說明

### 數據處理
- **augment_images.py**  
  做資料增強  

- **samplers.py**  
  使用 PyTorch 的 `WeightedRandomSampler`，針對多標籤數據不平衡進行加權採樣，  
  使少數類樣本更常被抽取，改善類別偏斜問題  

- **siggnet_common.py**  
  YOLO 目標檢測、SAM (Segment Anything Model) 分割  
  與舌頭圖像「旋正 / 對齊 / 方形化」(TIUO)  
  適合中醫舌診等醫學影像分析的多標籤多階段處理框架  

- **dataset.py**  
  修改後的資料集處理模組  

---

### 模型架構
- **httn.py**  
  定義頭到尾轉換網路（HTTN）的分類頭，解決長尾多標籤問題  
  包含：
  - `HTTNBaseHead`  
  - `HTTNEnsembleHead`  
  適用於醫療 AI 多標籤分類  

- **model.py**  
  使用 **ResNet50** 作為特徵提取骨幹，進行特徵降維  
  並結合 **HTTN**，適合不平衡數據 / 長尾分類任務  

- **mobilenetv3.py**  
  官方來源：[PyTorch Vision MobileNetV3](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py)  

- **resnet.py**  
  曹做的 ResNet50 實驗腳本  

---

### 損失函數
- **loss.py**  
  BCE loss function  

---

### 訓練與驗證
- **train.py**  
  修改過後的主訓練程式  

- **train_multibranch_tongue.py**  
  多分支舌頭分類訓練  

- **two_stage_train.py**  
  曹做的二階段驗證  

- **eval.py**  
  修改過後的驗證程式  

- **resnet_evaluate.py**  
  曹做的 ResNet 驗證  

- **simple_predict.py**  
  HTTN 驗證測試程式  

---

### 評估與混淆矩陣
- **confusion.py**  
  繪製混淆矩陣  

- **cm_builder.py**  
  根據多標籤 CSV，計算並異構 8 類單標籤混淆矩陣  

---

### 權重檔案
- **mobile_sam.pt**  
  從官方下載的 SAM 權重  

- **yolov8n.pt**  
  從官方下載的 YOLOv8n 權重  




