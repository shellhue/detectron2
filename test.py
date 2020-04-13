import torch

# class Exp(torch.Function):
#     @staticmethod
#     def forward(ctx, i):
#         result = i.exp()
#         ctx.save_for_backward(result)
#         return result

#     @staticmethod
#     def backward(ctx, grad_output):
#         result, = ctx.saved_tensors
#         return grad_output * result
    


def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook

x = torch.tensor([2.0], requires_grad=True)
y = 9 * x # 18  576 * 9 
z = y * 4 # 72   144 * 4 = 576
m = z * z # 5184  2 * z = 144
x.register_hook(set_grad(x))
y.register_hook(set_grad(y))
z.register_hook(set_grad(z))
m.register_hook(set_grad(m))
m.backward()
print(x.grad, y.grad, z.grad, m.grad)
m.backward()
print(x.grad, y.grad, z.grad, m.grad)