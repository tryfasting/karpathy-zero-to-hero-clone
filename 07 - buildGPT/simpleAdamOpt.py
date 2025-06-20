import torch

class SimpleAdam:
    def __init__(self, params, lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8):
        """
        Simple Adam optimizer Implementation

        Args:
            params : Parameters to optimize
            lr : Learning rate
            betas : (beta1, beta2) - momentum coefficients
            eps : Small value for numerical stability
        """
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_count = 0

        # Store state for each parameter
        self.state = {}
        for param in self.params:
            """왜 딕셔너리를 중첩해서 관리하는가?"""
            self.state[param] = {
                'm': torch.zeros_like(param.data), # 1st moment (exponential moving average of gradients)
                'v': torch.zeros_like(param.data), # 2nd moment (exponential moving average of squared gradients)
            }

    def zero_grad(self):
        """Initialize all parameter gradients to zero"""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self):
        """Core method that performs actual parameter updates"""
        self.step_count += 1

        for param in self.params:
            if param.grad is None:
                continue

            # Get current parameter's gradient and state
            grad = param.grad.data
            state = self.state[param]
            m, v = state['m'], state['v']

            # Step 1: Update 1st moment (exponential moving average of gradients)
            m.mul_(self.beta1).add_(grad, alpha=1-self.beta1)

            # Step 2: Update 2nd moment (exponential moving average of squared gradients)
            v.mul_(self.beta2).addcmul_(grad, grad, value=1-self.beta2)
            
            # Step 3: Bias Correction (correct bias toward zero in early steps)
            bias_correction1 = 1 - self.beta1 ** self.step_count
            bias_correction2 = 1 - self.beta2 ** self.step_count
            
            # Step 4: Calculate corrected moments
            corrected_m = m / bias_correction1
            corrected_v = v / bias_correction2
            
            # Step 5: Update parameters
            # param = param - lr * corrected_m / (sqrt(corrected_v) + eps)
            param.data.addcdiv_(corrected_m, corrected_v.sqrt().add_(self.eps), value=-self.lr)

#------------------------------------------------------------------------------------

# Actual usage example
def demonstrate_adam_optimizer():
    # Create simple linear model
    model = torch.nn.Linear(2, 1)
    criterion = torch.nn.MSELoss()
    
    # Use our custom Adam optimizer
    optimizer = SimpleAdam(model.parameters(), lr=0.01)
    
    # Sample data
    x = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y = torch.tensor([[3.0], [5.0], [7.0]])
    
    print("=== Adam Optimizer Operation Process ===")
    print(f"Initial weight: {model.weight.data}")
    print(f"Initial bias: {model.bias.data}")
    
    for epoch in range(3):
        print(f"\n--- Epoch {epoch + 1} ---")
        
        # Forward pass
        output = model(x)
        loss = criterion(output, y)
        print(f"Loss: {loss.item():.4f}")
        
        # Backward pass
        optimizer.zero_grad()  # Initialize gradients
        loss.backward()        # Calculate gradients
        
        # Print state before optimizer step
        print(f"Weight gradient: {model.weight.grad.data}")
        print(f"Bias gradient: {model.bias.grad.data}")
        
        # Parameter update
        optimizer.step()       # Update parameters
        
        print(f"Updated weight: {model.weight.data}")
        print(f"Updated bias: {model.bias.data}")

#------------------------------------------------------------------------------------

# Execute

if __name__ == "__main__":
    demonstrate_adam_optimizer()
    