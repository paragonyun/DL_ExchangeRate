import torch
import torch.nn as nn

from tqdm import tqdm

class Trainer:
    def __init__(
                self, 
                model, 
                loader,
                epoches,
                optimizer, 
                criterion,
                device="cuda" if torch.cuda.is_available() else "cpu"
                ):
        self.model = model
        self.loader = loader
        self.epoches = epoches
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
    
    def doit(self):
        print("🚀Start Training...🚀")
        print(f"Using Resource : {self.device}")
        self.model.train()
        for epoch in range(1, self.epoches + 1):
            running_loss = 0.0

            for x, y in tqdm(self.loader):
                x, y = x.to(self.device).float(), y.to(self.device).float()

                self.optimizer.zero_grad()
                torch.autograd.set_detect_anomaly(True)

                preds = self.model(inputs=x,
                                    target_len=7) # OW가 들어갑니다

                loss = self.criterion(preds, y)

                loss.backward()


                self.optimizer.step()

                running_loss += loss.item()
                
            
            print("============================================")
            print(running_loss)
            print("============================================")


            
            train_loss = running_loss / len(self.loader)

            print("==================================================")
            print(f'EPOCH [{epoch}/{self.epoches}]')
            print(f"LOSS : {train_loss:.3f}")
        
        print("📃 Save the Trained Model for Prediction...")
        torch.save(self.model.state_dict(), "./Final_Model.pth")
        print("✅ Done!")