include("plot_utils.jl")

# xs = (-10:0.05:10)
# ys = (-10:0.05:10)
# N = length(xs)
# xs, ys = xs' .* ones(N), ones(N)' .* ys
# ds = xs.^2 + ys.^2
# loss = exp.(-0.5*ds/4^2)

# fig = figure()
# ax = fig.add_subplot(projection="3d")
# ax.plot_surface(xs, ys, loss, cmap = "coolwarm", alpha = 1, linewidth=1, antialiased=false)
# #imshow(loss, cmap = "coolwarm")
# ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
# ax.axis("off")
# #ax.set_xlim(-10, 10)
# #ax.set_ylim(-10, 10)

# #x, y, dx, dy
# soa = [[5 5 0 0 0 1]; [5 -5 0 0 0 1]]
# X, Y, Z, dX, dY, dZ = [soa[:, i] for i = 1:size(soa, 2)]
# ax.quiver(X,Y,Z,dX,dY,dZ, color = "r")

# savefig("./figs/pg_schematic.png", bbox_inches = "tight")
# close()


using PyPlot
using PyCall
@pyimport matplotlib.patches as patch
fig = figure(figsize = (5,5))
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.00, right=1.00, bottom = 0.0, top = 1.0, wspace=0.35)
ax = fig.add_subplot(grids[1,1])

p_bbox = patch.FancyBboxPatch((0, 0.1), 1, 0.9,
                    boxstyle="Round,pad=0.00,rounding_size=0.1",
                    ec="k", fc="grey", alpha = 0.3
                    )
ax.add_patch(p_bbox)

p_bbox2 = patch.FancyBboxPatch((0, -1.0), 1, 0.9,
                    boxstyle="Round,pad=0.00,rounding_size=0.1",
                    ec="k", fc="grey", alpha = 0.3
                    )
ax.add_patch(p_bbox2)

x1, x2, x3 = 0.15, 0.5, 0.85
dx, dy = 0.6, 0.3
y1, y2, y3 = 0.25, 0.35, 0.8
size = 20
hl, hw, lw = 0.05,0.04,3

function draw_arrow(x,y,dx,dy)
    plt.arrow(x, y, dx, dy, color = "k", head_length = hl, head_width = hw, lw = lw, length_includes_head = true)
end
function text(x, y, s)
    plt.text(x, y, s, ha = "center", va = "center", size = size)
end

text(x1, y1, L"$h_1$")
text(x3, y1, L"$h_2$")
text(x2, y2, L"$\Delta h^\mathrm{PG}$")
text(x2, y3, L"$\hat{\tau}, R_{\hat{\tau}}$")
draw_arrow(x1+0.05, y1, dx, 0)
draw_arrow(x2, y3-0.05, 0, -dy)

text(x1, -y1, L"$h_1$")
text(x3, -y1, L"$h_2$")
text(x2, -y2, L"$\Delta h^\mathrm{RNN}$")
text(x1, -y3, L"$\hat{\tau}, R_{\hat{\tau}}$")
text(x3, -y3, L"$x_f$")
draw_arrow(x1+0.05, -y1, dx, 0)
draw_arrow(x1+0.05, -y3, dx, 0)
draw_arrow(x1, -y1-0.05, 0, -dy)
draw_arrow(x3, -y3+0.05, 0, dy)

plt.text(-0.02, 0.5, "policy gradient", rotation = 90, ha = "right", va = "center", size = size*1.0)
plt.text(-0.02, -0.5, "RNN dynamics", rotation = 90, ha = "right", va = "center", size = size*1.0)

ax.set_xlim(-0.2, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.axis("off")

savefig("./figs/pg_alpha_schematic.png", bbox_inches = "tight")
close()
