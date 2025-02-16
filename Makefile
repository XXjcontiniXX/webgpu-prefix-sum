DAWN_DIR = dawn/out/Debug/install

all:
	g++ main.cpp -o my_program -I$(DAWN_DIR)/include -L$(DAWN_DIR)/lib -Wl,-rpath=$(DAWN_DIR)/lib -lwebgpu_dawn -ldl -lpthread

clean:
	rm -f my_program