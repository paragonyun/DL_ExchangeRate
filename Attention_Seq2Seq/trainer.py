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
        device="cuda" if torch.cuda.is_available() else "cpu",
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

        best_loss = 10000

        fin_atten_weights = []

        for epoch in range(1, self.epoches + 1):
            running_loss = 0.0

            for x, y in tqdm(self.loader):
                x, y = x.to(self.device).float(), y.to(self.device).float()

                self.optimizer.zero_grad()
                # torch.autograd.set_detect_anomaly(True)
                
                preds, attn_weights_lst = self.model(inputs=x, target_len=7)#.to(self.device)  # OW가 들어갑니다

                preds = preds.to(self.device)

                loss = self.criterion(preds, y)

                loss.backward()

                self.optimizer.step()

                running_loss += loss.item()
                fin_atten_weights.append(attn_weights_lst)

            train_loss = running_loss / len(self.loader)

            print("==================================================")
            print(f"EPOCH [{epoch}/{self.epoches}]")
            print(f"TOTAL LOSS : {running_loss:3f}\tAVG LOSS : {train_loss:.3f}")

            if running_loss < best_loss:
                print("🚩 Saving Best Model...")
                torch.save(self.model.state_dict(), "./BEST_MODEL.pth")
                best_loss = running_loss

        print("📃 Save the Trained Model for Prediction...")
        torch.save(self.model.state_dict(), "./Final_Model.pth")
        print("✅ Done!")

        return fin_atten_weights