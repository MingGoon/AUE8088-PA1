import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

model_names  = ['AlexNet', 'AlexNet-Modified_1', 'AlexNet-Modified_2', 'ResNet-18', 'ResNet-34', 'EfficientNet-B0']
flops_mill   = [189.7, 1504.8, 1504.8, 296.3, 598.3, 64.5]      # 단위: MFLOPs
val_acc      = [0.291, 0.386, 0.517, 0.367, 0.355, 0.392]      # 단위: (0~1)


# 1) 파라미터 수 기준 플롯
plt.figure(figsize=(6,4))
plt.scatter(flops_mill, [a*100 for a in val_acc], s=50)      
for x, y, label in zip(flops_mill, val_acc, model_names):
    plt.text(x, y*100 + 0.5, label.upper(), ha='center')

plt.xlabel('Parameters (Millions)')
plt.ylabel('Validation Top-1 Accuracy (%)')
plt.title('Size–Accuracy Trade-Off')
plt.grid(True)
plt.tight_layout()
plt.show()


# 2) FLOPs 기준 플롯
plt.figure(figsize=(6,4))
plt.scatter(flops_mill, [a*100 for a in val_acc], s=50, color='tab:orange') 
for x, y, label in zip(flops_mill, val_acc, model_names):
    plt.text(x, y*100 + 0.5, label.upper(), ha='center')

plt.xlabel('FLOPs (Millions)')
plt.ylabel('Validation Top-1 Accuracy (%)')
plt.title('FLOPs–Accuracy Trade-Off')
plt.grid(True)
plt.margins(x=0.2, y=0.2)
plt.tight_layout()
plt.savefig('size_accuracy.png')