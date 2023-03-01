import torch
import torch.nn as nn

from tqdm import tqdm
import pickle


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

        total_atten_weights = torch.zeros(32, 14).to(self.device)

        for epoch in range(1, self.epoches + 1):
            running_loss = 0.0
            epoch_atten_weights = torch.zeros(32, 14).to(self.device)
            for x, y in tqdm(self.loader):
                x, y = x.to(self.device).float(), y.to(self.device).float()

                self.optimizer.zero_grad()
                # torch.autograd.set_detect_anomaly(True)
                
                preds, atten_weights = self.model(inputs=x, target_len=7)#.to(self.device)  # OW가 들어갑니다

                preds = preds.to(self.device)

                loss = self.criterion(preds, y)

                loss.backward(retain_graph=True)

                self.optimizer.step()

                running_loss += loss.item()
                
                if atten_weights.size(0) == 28:
                    temp_tensor = torch.zeros(4, 14).to(self.device)
                    atten_weights = torch.cat((atten_weights, temp_tensor), dim=0)

                epoch_atten_weights += atten_weights

            total_atten_weights += epoch_atten_weights

            total_atten_weights_dist = torch.softmax(total_atten_weights, dim=1)

            train_loss = running_loss / len(self.loader)

            print("======================================================================")
            print(f"EPOCH [{epoch}/{self.epoches}]")
            print(f"TOTAL LOSS : {running_loss:4f}\tAVG LOSS : {train_loss:.4f}")

            if epoch % 100 == 0:
                print("Attention Weights Distributions")
                print(total_atten_weights)
                print(total_atten_weights_dist)
                with open(f"./Attention_Weights_ep{epoch}.pkl", "wb") as f:
                    pickle.dump(total_atten_weights, f)

                with open(f"./Attention_Distribution_ep{epoch}.pkl", "wb") as f:
                    pickle.dump(total_atten_weights_dist, f)
                
                print("👍 Attention Weight and Distribusion is Saved")

                del total_atten_weights
                del total_atten_weights_dist
                total_atten_weights = torch.zeros(32, 14).to(self.device)

            if running_loss < best_loss:
                print("🚩 Saving Best Model...")
                torch.save(self.model.state_dict(), "./BEST_MODEL.pth")
                best_loss = running_loss

        print("📃 Save the Trained Model for Prediction...")
        torch.save(self.model.state_dict(), "./Final_Model.pth")
        print("✅ Done!")

        return total_atten_weights