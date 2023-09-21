import torch
import numpy as np

from examples.surface_fitting import chamfer_distance_two_side
# from examples.test.u_test import hausdorff_distance
torch.manual_seed(120)
from tqdm import tqdm
# from pytorch3d.loss import chamfer_distance
from NURBSDiff.nurbs_eval import SurfEval
from NURBSDiff.surf_eval import SurfEval as SurfEvalBS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from geomdl import exchange, utilities
from geomdl.visualization import VisMPL
from geomdl import compatibility
# import offset_eval as off

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', family='sans-serif') 
plt.rc('font', serif='Times') 
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def read_weights(filename, sep=","):
    try:
        with open(filename, "r") as fp:
            content = fp.read()
            content_arr = [float(w) for w in (''.join(content.split())).split(sep)]
            return content_arr
    except IOError as e:
        print("An error occurred: {}".format(e.args[-1]))
        raise e

def main():
    timing = []
    case = 'Ducky'
    if case=='Ducky':
        num_ctrl_pts1 = 14
        num_ctrl_pts2 = 13
        num_eval_pts_u = 64
        num_eval_pts_v = 64
        # knot_u = utilities.generate_knot_vector(3, 14)
        # knot_v = utilities.generate_knot_vector(3, 13)
        inp_ctrl_pts = torch.nn.Parameter(torch.rand(1,num_ctrl_pts1, num_ctrl_pts2, 3))
        knot_u = np.array([-1.5708, -1.5708, -1.5708, -1.5708, -1.0472, -0.523599, 0, 0.523599, 0.808217,
                              1.04015, 1.0472, 1.24824, 1.29714, 1.46148, 1.5708, 1.5708, 1.5708, 1.5708])
        knot_u = (knot_u - knot_u.min())/(knot_u.max()-knot_u.min())
        knot_v = np.array([-3.14159, -3.14159, -3.14159, -3.14159, -2.61799, -2.0944, -1.0472, -0.523599,
                              6.66134e-016, 0.523599, 1.0472, 2.0944, 2.61799, 3.14159, 3.14159, 3.14159, 3.14159])
        knot_v = (knot_v - knot_v.min())/(knot_v.max()-knot_v.min())
        # print(knot_u)
        ctrlpts = np.array(exchange.import_txt("./Ducky/duck1.ctrlpts", separator=" "))
        weights = np.array(read_weights("./Ducky/duck1.weights")).reshape(num_ctrl_pts1 * num_ctrl_pts2,1)
        target_ctrl_pts = torch.from_numpy(np.concatenate([ctrlpts,weights],axis=-1)).view(1,num_ctrl_pts1,num_ctrl_pts2,4)
        target_eval_layer = SurfEvalBS(num_ctrl_pts1, num_ctrl_pts2, knot_u=knot_u, knot_v=knot_v, dimension=3, p=3, q=3, out_dim_u=num_eval_pts_u, out_dim_v=num_eval_pts_v)
        target = target_eval_layer(target_ctrl_pts).float().cuda()

        # PTS = target.detach().numpy().squeeze()
        # Max_size = off.Max_size(np.reshape(PTS, [1, num_eval_pts_u * num_eval_pts_v, 3]))
        inp_ctrl_pts = torch.nn.Parameter(torch.rand((1,num_ctrl_pts1,num_ctrl_pts2,3), requires_grad=True).float().cuda())

    elif case=='Shark':
        surf_list = exchange.import_json("Shark/shark_solid.json")
        num_ctrl_pts1 = len(surf_list[0].knotvector_u) - surf_list[0].order_u
        num_ctrl_pts2 = len(surf_list[0].knotvector_v) - surf_list[0].order_v
        num_eval_pts_u = 512
        num_eval_pts_v = 512
        knot_u = np.array(surf_list[0].knotvector_u)
        knot_v = np.array(surf_list[0].knotvector_v)
        ctrlpts = np.array(surf_list[0].ctrlpts).reshape(num_ctrl_pts1,num_ctrl_pts2,3)
        weights = np.array(surf_list[0].weights).reshape(num_ctrl_pts1,num_ctrl_pts2,1)
        target_ctrl_pts = torch.from_numpy(np.concatenate([ctrlpts,weights],axis=-1)).view(1,num_ctrl_pts1,num_ctrl_pts2,4)
        target_eval_layer = SurfEvalBS(num_ctrl_pts1, num_ctrl_pts2, knot_u=knot_u, knot_v=knot_v, dimension=3, p=3, q=3, out_dim_u=num_eval_pts_u, out_dim_v=num_eval_pts_v)
        target = target_eval_layer(target_ctrl_pts).float().cuda()
        inp_ctrl_pts = torch.nn.Parameter(torch.rand((1,num_ctrl_pts1,num_ctrl_pts2,4), requires_grad=True).float().cuda())

    p = 3
    q = 3
    knot_int_u = torch.nn.Parameter(torch.ones(num_ctrl_pts1+p+1-2*p-1).unsqueeze(0).cuda(), requires_grad=True)
    # knot_int_u.data[0,3] = 0.0
    knot_int_v = torch.nn.Parameter(torch.ones(num_ctrl_pts2+q+1-2*q-1).unsqueeze(0).cuda(), requires_grad=True)
    # knot_int_v.data[0,3] = 0.0
    weights = torch.nn.Parameter(torch.ones(1,num_ctrl_pts1, num_ctrl_pts2, 1).cuda(), requires_grad=True)

    layer = SurfEval(num_ctrl_pts1, num_ctrl_pts2, dimension=3, p=3, q=3, out_dim_u=num_eval_pts_u, out_dim_v=num_eval_pts_v, method='tc', dvc='cuda').cuda()
    opt1 = torch.optim.LBFGS(iter([inp_ctrl_pts, weights]), lr=0.5, max_iter=3)
    opt2 = torch.optim.SGD(iter([knot_int_u, knot_int_v]), lr=1e-3)
    pbar = tqdm(range(1000))

    for i in pbar:
        # torch.cuda.empty_cache()
        knot_rep_p_0 = torch.zeros(1,p+1).cuda()
        knot_rep_p_1 = torch.zeros(1,p).cuda()
        knot_rep_q_0 = torch.zeros(1,q+1).cuda()
        knot_rep_q_1 = torch.zeros(1,q).cuda()


        def closure():
            opt1.zero_grad()
            opt2.zero_grad()
            # out = layer(inp_ctrl_pts)
            out = layer((torch.cat((inp_ctrl_pts,weights), -1), torch.cat((knot_rep_p_0,knot_int_u,knot_rep_p_1), -1), torch.cat((knot_rep_q_0,knot_int_v,knot_rep_q_1), -1)))
            loss = ((target-out)**2).mean() 
            # + chamfer_distance_two_side(out, target)
            # + 10 * hausdorff_distance(out, target) 
            # + 0.1 * laplacian_smoothing(inp_ctrl_pts)

            # out = out.reshape(1, num_eval_pts_u*num_eval_pts_v, 3)
            # tgt = target.reshape(1, num_eval_pts_u*num_eval_pts_v, 3)
            # loss = chamfer_distance_two_side(out,tgt)
            loss.backward(retain_graph=True)
            return loss

        if (i%300) < 30:
            loss = opt1.step(closure)
        else:
            loss = opt2.step(closure)        # with torch.no_grad():
        #     inp_ctrl_pts[:,0,:,:] = (inp_ctrl_pts[:,0,:,:]).mean(1)
        #     inp_ctrl_pts[:,-1,:,:] = (inp_ctrl_pts[:,-1,:,:]).mean(1)
        #     inp_ctrl_pts[:,:,0,:] = inp_ctrl_pts[:,:,-1,:] = (inp_ctrl_pts[:,:,0,:] + inp_ctrl_pts[:,:,-1,:])/2

        out = layer((torch.cat((inp_ctrl_pts,weights), -1), torch.cat((knot_rep_p_0,knot_int_u,knot_rep_p_1), -1), torch.cat((knot_rep_q_0,knot_int_v,knot_rep_q_1), -1)))
        target = target.reshape(1,num_eval_pts_u,num_eval_pts_v,3)
        out = out.reshape(1,num_eval_pts_u,num_eval_pts_v,3)

        if loss.item() < 1e-4:
            break
        pbar.set_description("Loss %s: %s" % (i+1, loss.item()))
        # print(knot_int_u)
        # print(knot_int_v)
    train_uspan_uv, train_vspan_uv = layer.getuvspan()
    target_uspan_uv, target_vspan_uv = target_eval_layer.getuvsapn()

    def generate_gradient(start_color, end_color, steps):
        # Convert the start and end colors to RGB tuples
        rgb_start = tuple(int(start_color[i:i+2], 16) for i in (1, 3, 5))
        rgb_end = tuple(int(end_color[i:i+2], 16) for i in (1, 3, 5))
        # rgb_inter = tuple(int(intermediate_color[i:i+2], 16) for i in (1, 3, 5))
        # Calculate the step size for each RGB component
        step_size = tuple((rgb_end[i] - rgb_start[i]) / (steps - 1) for i in range(3))
        # step_size_b = tuple((rgb_end[i] - rgb_inter[i]) / (steps / 2 - 1) for i in range(3))
        # Generate the gradient colors
        gradient = []
        for i in range(int(steps)):
            # Calculate the RGB values for the current step
            r = int(rgb_start[0] + i * step_size[0])
            g = int(rgb_start[1] + i * step_size[1])
            b = int(rgb_start[2] + i * step_size[2])
            
            # Convert the RGB values to a hexadecimal string
            hex_color = '#' + format(r, '02x') + format(g, '02x') + format(b, '02x')
            
            # Add the hexadecimal color string to the gradient list
            gradient.append(hex_color)
        return gradient
    colors = generate_gradient('#ff0000', '#00ff00', (num_ctrl_pts1 - 3) * (num_ctrl_pts2 - 3) // 2) + generate_gradient('#00ff00', '#0000ff', (num_ctrl_pts1 - 3) * (num_ctrl_pts2 - 3) // 2)


    fig = plt.figure(figsize=(15, 4))
    ax1 = fig.add_subplot(131, projection='3d', adjustable='box', proj_type='ortho')
    target_mpl = target.cpu().numpy().squeeze()
    with open('./test/generated/test.off', 'w') as f:
        # Loop over the array rows
        x = target_mpl
        x = x.reshape(-1, 3)
        
        for i in range(num_eval_pts_u * num_eval_pts_v):
                # print(predicted_target[i, j, :])
                line = str(x[i, 0]) + ' ' + str(x[i, 1]) + ' ' + str(x[i, 2]) + '\n'
                f.write(line)
               
    predicted = out.detach().cpu().numpy().squeeze()
    ctrlpts = ctrlpts.reshape(num_ctrl_pts1, num_ctrl_pts2, 3)
    predctrlpts = inp_ctrl_pts.detach().cpu().numpy().squeeze()
    # predctrlpts = predctrlpts[:, :, :3] / predctrlpts[:, :, 3:]
    u_index = 0
    for i in range(num_ctrl_pts1 - 3):
        u_index += target_uspan_uv[i + 3]
        v_index = 0
        for j in range(num_ctrl_pts2 - 3):
            if u_index == 512 or v_index == 512 or u_index - target_uspan_uv[i + 3] == 0 or u_index - target_uspan_uv[i + 3] == 256:
                if u_index == 512: 
                    ax1.plot_wireframe(target_mpl[u_index - target_uspan_uv[i + 3]:-1, v_index:v_index + train_vspan_uv[j + 3], 0],
                                target_mpl[u_index - target_uspan_uv[i + 3]:-1, v_index:v_index + train_vspan_uv[j + 3], 1],
                                target_mpl[u_index - target_uspan_uv[i + 3]:-1, v_index:v_index + train_vspan_uv[j + 3], 2],
                                    color=colors[i * (num_ctrl_pts2 - 3) + j], label = 'u = 1, v = 1' if(v_index + train_vspan_uv[j + 3] == 512) else None)
                elif v_index == 512:
                    ax1.plot_wireframe(target_mpl[u_index - target_uspan_uv[i + 3]:-1, v_index:-1, 0],
                                target_mpl[u_index - target_uspan_uv[i + 3]:-1, v_index:-1, 1],
                                target_mpl[u_index - target_uspan_uv[i + 3]:-1, v_index:-1, 2],
                                    color=colors[i * (num_ctrl_pts2 - 3) + j])
                elif u_index - target_uspan_uv[i + 3] == 0:
                    ax1.plot_wireframe(target_mpl[u_index - target_uspan_uv[i + 3]:u_index, v_index:v_index + train_vspan_uv[j + 3], 0],
                                    target_mpl[u_index - target_uspan_uv[i + 3]:u_index, v_index:v_index + train_vspan_uv[j + 3], 1],
                                    target_mpl[u_index - target_uspan_uv[i + 3]:u_index, v_index:v_index + train_vspan_uv[j + 3], 2],
                                    color=colors[i * (num_ctrl_pts2 - 3) + j], label = 'u = 0, v = 0' if(v_index == 0) else None)
                elif u_index - target_uspan_uv[i + 3] == 256:
                    ax1.plot_wireframe(target_mpl[u_index - target_uspan_uv[i + 3]:u_index, v_index:v_index + train_vspan_uv[j + 3], 0],
                                target_mpl[u_index - target_uspan_uv[i + 3]:u_index, v_index:v_index + train_vspan_uv[j + 3], 1],
                                target_mpl[u_index - target_uspan_uv[i + 3]:u_index, v_index:v_index + train_vspan_uv[j + 3], 2],
                                    color=colors[i * (num_ctrl_pts2 - 3) + j], label = 'u = 0.5, v = 0.5'.format(v_index + train_vspan_uv[j + 3]) if(v_index + train_vspan_uv[j + 3] == 256) else None)
            else:
                ax1.plot_wireframe(target_mpl[u_index - target_uspan_uv[i + 3]:u_index, v_index:v_index + train_vspan_uv[j + 3], 0],
                                target_mpl[u_index - target_uspan_uv[i + 3]:u_index, v_index:v_index + train_vspan_uv[j + 3], 1],
                                target_mpl[u_index - target_uspan_uv[i + 3]:u_index, v_index:v_index + train_vspan_uv[j + 3], 2],
                                    color=colors[i * (num_ctrl_pts2 - 3) + j])
             
            v_index += train_vspan_uv[j + 3]
    ax1.plot_wireframe(ctrlpts[:, :, 0], ctrlpts[:, :, 1], ctrlpts[:, :, 2], linestyle='dashed', color='orange',
                               label='Target Control Points')
    # ax1.set_zlim(-1,3)
    # ax1.set_xlim(-1,4)
    # ax1.set_ylim(-2,2)
    ax1.azim = 45
    ax1.dist = 6.5
    ax1.elev = 30
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1._axis3don = False
    # ax.legend(loc='upper left')
    ax2 = fig.add_subplot(132, projection='3d', adjustable='box')
    u_index = 0
    for i in range(num_ctrl_pts1 - 3):
        u_index += train_uspan_uv[i + 3]
        v_index = 0
        for j in range(num_ctrl_pts2 - 3):
            if u_index == 512 or v_index == 512 or u_index - train_uspan_uv[i + 3] == 0 or u_index - train_uspan_uv[i + 3] == 256:
                if u_index == 512: 
                    ax2.plot_wireframe(target_mpl[u_index - train_uspan_uv[i + 3]:-1, v_index:v_index + target_vspan_uv[j + 3], 0],
                                target_mpl[u_index - train_uspan_uv[i + 3]:-1, v_index:v_index + target_vspan_uv[j + 3], 1],
                                target_mpl[u_index - train_uspan_uv[i + 3]:-1, v_index:v_index + target_vspan_uv[j + 3], 2],
                                    color=colors[i * (num_ctrl_pts2 - 3) + j], label = 'u = 1, v = 1' if(v_index + target_vspan_uv[j + 3] == 512) else None)
                elif v_index == 512:
                    ax2.plot_wireframe(target_mpl[u_index - train_uspan_uv[i + 3]:-1, v_index:-1, 0],
                                target_mpl[u_index - train_uspan_uv[i + 3]:-1, v_index:-1, 1],
                                target_mpl[u_index - train_uspan_uv[i + 3]:-1, v_index:-1, 2],
                                    color=colors[i * (num_ctrl_pts2 - 3) + j])
                elif u_index - train_uspan_uv[i + 3] == 0:
                    ax2.plot_wireframe(target_mpl[u_index - train_uspan_uv[i + 3]:u_index, v_index:v_index + target_vspan_uv[j + 3], 0],
                                    target_mpl[u_index - train_uspan_uv[i + 3]:u_index, v_index:v_index + target_vspan_uv[j + 3], 1],
                                    target_mpl[u_index - train_uspan_uv[i + 3]:u_index, v_index:v_index + target_vspan_uv[j + 3], 2],
                                    color=colors[i * (num_ctrl_pts2 - 3) + j], label = 'u = 0, v = 0' if(v_index == 0) else None)
                elif u_index - train_uspan_uv[i + 3] == 256:
                    ax2.plot_wireframe(target_mpl[u_index - train_uspan_uv[i + 3]:u_index, v_index:v_index + target_vspan_uv[j + 3], 0],
                                target_mpl[u_index - train_uspan_uv[i + 3]:u_index, v_index:v_index + target_vspan_uv[j + 3], 1],
                                target_mpl[u_index - train_uspan_uv[i + 3]:u_index, v_index:v_index + target_vspan_uv[j + 3], 2],
                                    color=colors[i * (num_ctrl_pts2 - 3) + j], label = 'u = 0.5, v = 0.5'.format(v_index + target_vspan_uv[j + 3]) if(v_index + target_vspan_uv[j + 3] == 256) else None)
            else:
                ax2.plot_wireframe(target_mpl[u_index - train_uspan_uv[i + 3]:u_index, v_index:v_index + target_vspan_uv[j + 3], 0],
                                target_mpl[u_index - train_uspan_uv[i + 3]:u_index, v_index:v_index + target_vspan_uv[j + 3], 1],
                                target_mpl[u_index - train_uspan_uv[i + 3]:u_index, v_index:v_index + target_vspan_uv[j + 3], 2],
                                    color=colors[i * (num_ctrl_pts2 - 3) + j])
             
            v_index += target_vspan_uv[j + 3]
    ax2.plot_wireframe(predctrlpts[:, :, 0], predctrlpts[:, :, 1], predctrlpts[:, :, 2], linestyle='dashed',
                               color='orange', label='Predicted Control Points')
    ax2.azim = 45
    ax2.dist = 6.5
    ax2.elev = 30
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax2._axis3don = False
    # ax.legend(loc='upper left')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_zlim(-1,3)
    # ax.set_xlim(-1,4)
    # ax.set_ylim(-2,2)
    ax3 = fig.add_subplot(133, adjustable='box')
    error_map = (((predicted - target_mpl) ** 2) / target_mpl).sum(-1)
    # im3 = ax.imshow(error_map, cmap='jet', interpolation='none', extent=[0,128,0,128])
    im3 = ax3.imshow(error_map, cmap='jet', interpolation='none', extent=[0, 128, 0, 128], vmin=-0.001, vmax=0.001)
    # fig.colorbar(im3, shrink=0.4, aspect=5)
    fig.colorbar(im3, shrink=0.4, aspect=5, ticks=[-0.001, 0, 0.001])
    ax3.set_xlabel('$u$')
    ax3.set_ylabel('$v$')
    x_positions = np.arange(0, 128, 20)  # pixel count at label position
    plt.xticks(x_positions, x_positions)
    plt.yticks(x_positions, x_positions)
    ax3.set_aspect(1)
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.tight_layout()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    # finally we invoke the legend (that you probably would like to customize...)

    fig.legend(lines, labels, ncol=2, loc='lower center', bbox_to_anchor=(0.33, 0.0), )
    plt.savefig('ducky_reparameterization.pdf') 
    plt.show()
    pass
    # layer_2 = SurfEval(num_ctrl_pts1, num_ctrl_pts2, knot_u=knot_u, knot_v=knot_v, dimension=3, p=3, q=3,
    #                    out_dim_u=512,
    #                    out_dim_v=512, dvc='cpp')
    # weights = torch.ones(1, num_ctrl_pts1, num_ctrl_pts2, 1)
    # out_2 = layer_2(torch.cat((inp_ctrl_pts, weights), axis=-1))

    # target_2 = target.view(1, num_eval_pts_u * num_eval_pts_v, 3)
    # out_2 = out_2.view(1, 512 * 512, 3)

    # loss, _ = chamfer_distance(target_2, out_2)

    # print('Max size is  ==  ', Max_size)
    # print('Chamber loss is   ===  ',  loss * 10000)

if __name__ == '__main__':
    main()
