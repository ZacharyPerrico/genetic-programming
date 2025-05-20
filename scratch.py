from node import *

# Collatz Conjecture
f = Node.if_then_else(
    x % 2,
    3 * x + 1,
    x / 2,
)

# f = 2/4 + 7/4 * x +  (-2/4 + -5/4 * x) * (-1)**x
# f = 2/4 + 7/4*x + (-2/4 + -5/4*x) * cos(pi * x)
# f(n) = 2 / 4 + 7 / 4 * f(n-1) + (-2 / 4 + -5 / 4 * f(n-1)) * cos(pi * f(n-1))
# i = 43
# y = [i]
# while i != 1:
#     print(i)
#     i = f(i)
#     y.append(i)
# y = np.array(y)
# x = np.arange(len(y))
# # Loop plot
# xx = y.copy()
# yy = y.copy()
# yy[0] = 0
# xx[1::2] = xx[0::2]
# yy[2::2] = yy[1:-1:2]
# plt.plot(xx, yy)
# plt.scatter(xx,yy)
# plt.axline((0, 1), (1, 4), ls=':')
# plt.axline((0, 0), (1, 1/2), ls=':')
# plt.axline((1, 0), (4, 1), ls=':')
# plt.axline((0, 0), (1/2, 1), ls=':')
# plt.plot()
# plt.show()
# f = (((((x-x)+(x+x))**((x+x)/x))*(x+x))/((x+x)-x))
# f = (((x+x)*((((((((x+x)*(x/x))*x)+(((x-(x*(x+x)))*x)+(x*x)))+((((x-((x*(((((x-x)-x)/x)+x)/x))/x))/x)*x)*x))+x)/((x-x)+x))-((x+x)*(0-x))))/((x+((x+(((((((((((x/((0/(x-x))*(((x+((x/x)-(x*x)))+(x+((x/x)*x)))*x)))+((x+x)/x))-(((((x/(x-x))+x)/(((((x+x)/x)-((x-x)+x))+x)/x))*x)*(x/x)))+x)+x)/x)-x)/(x*x))/x)-x)+x))/x))-x))
# f = ((((x+x)**(((((x/x)+x)+x)+x)/x))-x)/((x+x)-x))
# (((max((if_then_else(x,x,((x*(if_then_else((x|x),abs(x),x)|(abs(x)-(x+x))))|abs(x)))*x),abs(((((min(x,x)+((if_then_else(x,x,x)|(x-x))&x))+x)|x)+max((max((abs(((min(x,x)+max(((((x+min(x,x))|x)|x)+(x&x)),x))+(((((if_then_else(0,x,x)&(x/x))+min(if_then_else(x,x,x),min(x,x)))&(x+x))+abs(x))|(x|x))))+min(x,x)),x)+if_then_else((x-x),x,x)),x))))-x)-min(x,x))*x)


# s = 8
# n = Node.get_bits(x, 0, s)
# c = Node.get_bits(x, s, s) + 1

# # c=c n=n
# # n == 1
# # c=0 n=c
# # c == 0
# # c=0 n=0

# x = x/2-1
#
# next_x = Node.if_then_else(
#     x % 2,
#     3 * x + 1,
#     x / 2,
# )
#
# f = Node.if_then_else(
#     next_x == 1,
#     -1,
#     Node.if_then_else(
#         c == 0,
#         0,
#         c
#     ),
# )


# nn = Node.if_then_else(
#     n == 1,
#     c,
#     Node.if_then_else(
#         c == 0,
#         0,
#         Node.if_then_else(
#             n % 2,
#             3 * n + 1,
#             n / 2,
#         )
#     )
# )
# f = cc * 2**s + nn
