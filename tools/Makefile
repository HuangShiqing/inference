CROSS_COMPILE =#arm-linux-gnueabi-#aarch64-linux-gnu-#指定交叉编译器
DEBUG = 0#指定当前为debug模式
OPENMP = 1#指定使用openmp

CC = $(CROSS_COMPILE)gcc#指定编译器
CFLAGS = -I../src/ -I../src/darknet/ -I../src/gstreamer -I../src/gtk/ -I../src/dip/ -I../src/neon/ -Wall#指定头文件目录
LDFLAGS = #指定库文件目录
LIBS = -lm -lpthread#指定库文件名称
TARGET = quantize#最终生成的可执行文件名

VPATH = ../src/:../src/darknet/:../src/gtk/:../src/gstreamer/:../src/dip/:../src/neon/:./tools/#告诉makefile去哪里找依赖文件和目标文件

OBJ = quantize.o network.o utils.o im2col.o my_layer.o image.o model.o parser.o \
	  connected_layer.o convolutional_layer.o maxpool_layer.o avgpool_layer.o \
	  batchnorm_layer.o activations.o box.o route_layer.o upsample_layer.o yolo_layer.o \
	  blas.o gemm.o  convert.o draw.o#中间过程所涉及的.o文件	 

#选择debug还是release
ifeq ($(DEBUG), 1)
CFLAGS+=-O0 -g
else
CFLAGS+=-O1#TODO:使用Ofast会导致gemm_neon.s出现segmentation failt
endif
#选择是否使用openmp
ifeq ($(OPENMP), 1)
CFLAGS += -fopenmp
endif
 
OBJDIR = ./obj/#存放.o文件的文件夹
OBJS = $(addprefix $(OBJDIR), $(OBJ))#添加路径

#指定需要完成的编译的对象
all: obj $(TARGET)

#将所有的.o文件链接成最终的可执行文件，需要库目录和库名，注意，OBJS要在LIBS之前。另外，如果要指定.o的生成路径，需要保证TARGET的依赖项是含路径的
$(TARGET):$(OBJS)
		$(CC) $(OBJS) $(CFLAGS) $(LDFLAGS) $(LIBS) -o $(TARGET)
#这个不是静态模式，而是通配符，第一个%类似bash中的*。
$(OBJDIR)%.o: %.c
		$(CC) -c $(CFLAGS) $< -o $@			
$(OBJDIR)%.o: %.s
		$(CC) -c $(CFLAGS) $< -mfpu=neon -mcpu=cortex-a53 -o $@		

#用于生成存放.o文件的文件夹
obj:
		mkdir obj
.PHONY : clean
clean :#删除生成的文件夹
		-rm -r obj $(TARGET)