setup ;

% -------------------------------------------------------------------------
% Part 1.1: Linear convolution
% -------------------------------------------------------------------------

% Read an example image
% 384x512x3
x = imread('peppers.png') ;

% Convert to single format
% 相当于映射从255到1，转变为单精度
% https://www.mathworks.com/help/images/ref/im2single.html
x = im2single(x) ;

% Visualize the input x
% imagesc(x)Matlab会自动获取矩阵中数的最小值和最大值
% 并把区间[Cmin,Cmax]映射到colormap
figure(1) ; clf ; imagesc(x) ;

% Create a bank of linear filters
% 创建滤波器，WxLxCxN,单精度类型
w = randn(5,5,3,10,'single') ;

% Apply the convolutional operator
% Y = VL_NNCONV(X, F, B) computes the convolution of the image X
% with the filter bank F and biases B
% y: 380x508x10
y = vl_nnconv(x, w, []) ;

% Visualize the output y
% 上面10个滤波器，产生10张图片
figure(2) ; clf ; vl_imarraysc(y) ; colormap gray ;

% Try again, downsampling the output
% y_ds: 24x32x10
y_ds = vl_nnconv(x, w, [], 'stride', 16) ;
figure(3) ; clf ; vl_imarraysc(y_ds) ; colormap gray ;

% Try padding
% y_pad: 388x516x10
y_pad = vl_nnconv(x, w, [], 'pad', 4) ;
figure(4) ; clf ; vl_imarraysc(y_pad) ; colormap gray ;

% Manually design a filter
w = [0  1 0 ;
     1 -4 1 ;
     0  1 0 ] ;
w = single(repmat(w, [1, 1, 3])) ;
y_lap = vl_nnconv(x, w, []) ;

figure(5) ; clf ; colormap gray ;
% y_lap: 382x510x1
subplot(1,2,1) ; imagesc(y_lap) ; title('filter output') ;
% 负绝对值y_lap: 382x510x1
subplot(1,2,2) ; imagesc(-abs(y_lap)) ; title('- abs(filter output)') ;


% -------------------------------------------------------------------------
% Part 1.2: Non-linear gating (ReLU)
% -------------------------------------------------------------------------

w = single(repmat([1 0 -1], [1, 1, 3])) ;
% cat(x)函数不是很懂
w = cat(4, w, -w) ;
% y: 384x510x2
y = vl_nnconv(x, w, []);
figure; clf ; vl_imarraysc(y) ; colormap gray ;
z = vl_nnrelu(y) ;

figure(6) ; clf ; colormap gray ;
subplot(1,2,1) ; vl_imarraysc(y) ;
subplot(1,2,2) ; vl_imarraysc(z) ;

% -------------------------------------------------------------------------
% Part 1.2: Pooling
% -------------------------------------------------------------------------

y = vl_nnpool(x, 15) ;
% y: 370x498x3
figure(7) ; clf ; imagesc(y) ;

% -------------------------------------------------------------------------
% Part 1.3: Normalization
% -------------------------------------------------------------------------

rho = 5 ;
kappa = 0 ;
alpha = 1 ;
beta = 0.5 ;

% 维度不变
y_nrm = vl_nnnormalize(x, [rho kappa alpha beta]) ;
figure(8) ; clf ; imagesc(y_nrm) ;
