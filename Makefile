CROSS_COMPILE=aarch64-linux-gnu-
CC=$(CROSS_COMPILE)gcc
VPATH=./src/:./src/darknet/
CFLAGS =-W -g -O0  -I./src/ -I./src/darknet/
LDFLAGS = -lm#数学计算库
obj = 

target=inference
OBJDIR=./obj/
obj+= main.o network.o utils.o im2col.o parser.o list.o option_list.o my_layer.o image.o model.o connected_layer.o convolutional_layer.o maxpool_layer.o avgpool_layer.o batchnorm_layer.o activations.o blas.o gemm.o
objs=$(addprefix $(OBJDIR),$(obj))

$(target): $(obj)
	$(CC) $(CFLAGS) $(objs) -o inference $(LDFLAGS)

main.o:main.c
	$(CC) $(CFLAGS) -c $< -o $(OBJDIR)$@ 

image.o:image.c
	$(CC) $(CFLAGS) -c $< -o $(OBJDIR)$@ 
utils.o:utils.c
	$(CC) $(CFLAGS) -c $< -o $(OBJDIR)$@ 
im2col.o:im2col.c
	$(CC) $(CFLAGS) -c $< -o $(OBJDIR)$@
parser.o:parser.c
	$(CC) $(CFLAGS) -c $< -o $(OBJDIR)$@	 
#不知道干嘛的
list.o:list.c
	$(CC) $(CFLAGS) -c $< -o $(OBJDIR)$@
option_list.o:option_list.c
	$(CC) $(CFLAGS) -c $< -o $(OBJDIR)$@

network.o:network.c
	$(CC) $(CFLAGS) -c $< -o $(OBJDIR)$@ 
model.o:model.c
	$(CC) $(CFLAGS) -c $< -o $(OBJDIR)$@
my_layer.o:my_layer.c
	$(CC) $(CFLAGS) -c $< -o $(OBJDIR)$@ 
connected_layer.o:connected_layer.c
	$(CC) $(CFLAGS) -c $< -o $(OBJDIR)$@ 	
convolutional_layer.o:convolutional_layer.c
	$(CC) $(CFLAGS) -c $< -o $(OBJDIR)$@ 
maxpool_layer.o:maxpool_layer.c
	$(CC) $(CFLAGS) -c $< -o $(OBJDIR)$@ 
avgpool_layer.o:avgpool_layer.c
	$(CC) $(CFLAGS) -c $< -o $(OBJDIR)$@  
batchnorm_layer.o:batchnorm_layer.c
	$(CC) $(CFLAGS) -c $< -o $(OBJDIR)$@	
activations.o:activations.c
	$(CC) $(CFLAGS) -c $< -o $(OBJDIR)$@
blas.o:blas.c
	$(CC) $(CFLAGS) -c $< -o $(OBJDIR)$@	 		
gemm.o:gemm.c
	$(CC) $(CFLAGS) -c $< -o $(OBJDIR)$@
clean:
	rm -f $(objs) $(target)		
