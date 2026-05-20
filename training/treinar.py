from ultralytics import YOLO
import torch

def main():
    # VERIFICAÇÃO DE HARDWARE
    if torch.backends.mps.is_available():
        print("✅ Metal (MPS) detectado! Usando GPU do Mac.")
        device_choice = 'mps'
    else:
        print("⚠️ Metal não detectado. Usando CPU (CUIDADO: Lento e esquenta!)")
        device_choice = 'cpu'

    # Carrega o modelo Nano (O mais leve para evitar que seu Mac desligue)
    model = YOLO('yolov8n.pt')

    # INICIA O TREINO COM LIMITADORES
    model.train(
    data='data.yaml',
    epochs=50,
    imgsz=416,
    batch=8,
    workers=2,
    device= 'mps',
    cache=False,
    patience=15,
    optimizer='AdamW',
    lr0=0.001,
    cos_lr=True,
    amp=True,
    val= False
)

    model.val() # Validação final para ver o resultado
    
if __name__ == '__main__':
    main()