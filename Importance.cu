#include <stdio.h>
#include <stdint.h>

__constant__ int dc_filterX[3 * 3];
__constant__ int dc_filterY[3 * 3];

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	
	if (strcmp(type, "P3") != 0) // In this exercise, we don't touch other types
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);
	
	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	for (int i = 0; i < width * height; i++)
		fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

	fclose(f);
}

void writePnm(uchar3 * pixels, int width, int height, char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	fprintf(f, "P3\n%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height; i++)
		fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);
	
	fclose(f);
}

char * concatStr(const char * s1, const char * s2)
{
    char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

// Function to swap two numbers
void swap(char *x, char *y) 
{
  char t = *x; *x = *y; *y = t;
}

// Function to reverse `buffer[i…j]`
char* reverse(char *buffer, int i, int j)
{
	while (i < j) 
	{
		swap(&buffer[i++], &buffer[j--]);
	}
	return buffer;
}

// Iterative function to implement `itoa()` function in C
char* itoa(int value, char* buffer, int base)
{
	// invalid input
	if (base < 2 || base > 32) 
	{
		return buffer;
	}

	// consider the absolute value of the number
	int n = abs(value);

	int i = 0;
	while (n)
	{
		int r = n % base;

		if (r >= 10) 
		{
			buffer[i++] = 65 + (r - 10);
		}
		else 
		{
			buffer[i++] = 48 + r;
		}
		n = n / base;
	}

	// if the number is 0
	if (i == 0) 
	{
		buffer[i++] = '0';
	}

	// If the base is 10 and the value is negative, the resulting string
	// is preceded with a minus sign (-)
	// With any other base, value is always considered unsigned
	if (value < 0 && base == 10) 
	{
		buffer[i++] = '-';
	}

	buffer[i] = '\0'; // null terminate string

	// reverse the string and return it
	return reverse(buffer, 0, i - 1);
}

void printArray(int * a, int n)
{
    for (int i = 0; i < n; i++)
        printf("%i ", a[i]);
    printf("\n");
}

// By Host ---------------------------
void calculateImportance(uint8_t * grayPixels, int width, int height,
		int * filterX, int * filterY, int * importance)
{
	for (int R = 0; R < height; R++)
		for (int C = 0; C < width; C++)
		{
			int imp_x = 0, imp_y = 0;
			for (int filterR = 0; filterR < 3; filterR++)
				for (int filterC = 0; filterC < 3; filterC++)
				{
					int filterX_Val = filterX[filterR*3 + filterC];
					int filterY_Val = filterY[filterR*3 + filterC];
					
					int inPixelsR = R - 1 + filterR;
					int inPixelsC = C - 1 + filterC;
					inPixelsR = min(max(0, inPixelsR), height - 1);
					inPixelsC = min(max(0, inPixelsC), width - 1);

					int grayPixel = grayPixels[inPixelsR*width + inPixelsC];
					
					imp_x += filterX_Val * grayPixel;
					imp_y += filterY_Val * grayPixel;
				}
			importance[R*width + C] = sqrt(imp_x * imp_x + imp_y * imp_y); 
		}
}

void calculateDirection(int * importance, int width, int height)
{
	for (int R = height - 2; R >= 0; R--)
		for (int C = 0; C < width; C++)
		{
			int i = R*width + C;
			int i_lower = i + width;
			int dir = (C == width - 1) ? -1 : 1;
			
			// Find lower row's min 
			if (importance[i_lower] < importance[i_lower + dir])
				dir = 0;

			if (C > 0 && C < width - 1)
				if (importance[i_lower + dir] > importance[i_lower - 1])
					dir = -1;

			importance[i] += importance[i_lower + dir];
		}
}

void seamCarving(int * importance, 
			int width, int height, int loop, 
			uint8_t * grayPixels, uchar3 * inPixels, 
			char * buffer=nullptr, bool write=false)
{
	//-> find head of seam
	int min = 0;
	for (int i = 1; i < width; i++)
		if (importance[i] < importance[min])
			min = i;

	//-> find seam
	int * seam = (int *)malloc(height * sizeof(int));
	for (int r = 0; r < height; r++)
	{
		if (r == 0)
			seam[r] = min;
		else
		{
			int i = seam[r - 1];
			int C = i - (r - 1) * width;
			int dir = (C == width - 1) ? -1 : 1;
			
			// Find lower row's min 
			if (importance[i + width] < importance[i + width + dir])
				dir = 0;

			if (C > 0 && C < width - 1)
				if (importance[i + width + dir] > importance[i + width - 1])
					dir = -1;

			seam[r] = seam[r - 1] + width + dir;
		}
			
		inPixels[seam[r]] = make_uchar3(255,20,147);
	}

	// Write picture before carving
	if (write){
		itoa(loop, buffer, 10);
		char * fileSeam = concatStr("before_", buffer);
		writePnm(inPixels, width, height, concatStr(fileSeam, ".pnm"));
	}

	//-> remove seam
	int count = 0;
	for (int R = 0; R < height; R++)
		for (int C = 0; C < width; C++)
		{
			int i = R*width + C;
			if (i != seam[R])
			{
				inPixels[count] = inPixels[i];
				grayPixels[count++] = grayPixels[i];
			}
		}

	free(seam);
}

void seamCarvingbyHost(uchar3 * inPixels, int width, int height,
		uint8_t * outGrayPixels, uchar3 * outPixels, int * outImportance,
		int * filterX, int * filterY, int pixelsReduce
		)
{
	GpuTimer timer;
	timer.Start();

	// Gray Scale
	uint8_t * grayPixels= (uint8_t *)malloc(width * height * sizeof(uint8_t));
	for (int r = 0; r < height; r++)
		for (int c = 0; c < width; c++)
		{
			int i = r * width + c;
			outGrayPixels[i] = 0.299f*inPixels[i].x + 0.587f*inPixels[i].y + 0.114f*inPixels[i].z;
			grayPixels[i] = outGrayPixels[i];
		}

	// char buffer [25];
	// input reduce pixels 
	//using for loop ... 
	for (int loop = 0; loop < pixelsReduce; loop++)
	{
		// Calculate importance
		int * importance = (int *)malloc(width * height * sizeof(int));
		calculateImportance(grayPixels, width, height, filterX, filterY, importance);
		for (int R = 0; R < height; R++)
			for (int C = 0; C < width; C++)
			{
				int i = R*width +C;
				outImportance[i] = importance[i];
			}
		
		// Calculate directions
		calculateDirection(importance, width, height);

		// Find smallest seam and remove
		seamCarving(importance, width, height, loop, grayPixels, inPixels);

		// Reduce width every loop
		width--;

		free(importance);
	}

	// Write output
	for (int R = 0; R < height; R++)
		for (int C = 0; C < width; C++)
		{
			int i = R*width + C;
			outPixels[i] = inPixels[i];
		}

	free(grayPixels);

	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time (use host): %f ms\n\n", time);
}

// By Device ------------------------
// Gray Scale Kernel
__global__ void convertRgb2GrayKernel(uchar3 * inPixels, int width, int height, 
		uint8_t * outPixels)
{
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (r < height && c < width)
	{
		int i = r * width + c;
		outPixels[i] = 0.299f*inPixels[i].x + 0.587f*inPixels[i].y + 0.114f*inPixels[i].z;
	}
}

// Calculate importance kernel
__global__ void calculateImportanceKernel2(uint8_t * grayPixels, 
		int width, int height, int filterWidth, 
		int * importance)
{
// TODO
	int c= blockIdx.x * blockDim.x + threadIdx.x;
	int r= blockIdx.y * blockDim.y + threadIdx.y;

	if (r < height && c < width)
	{
		int imp_X = 0, imp_Y = 0;
		for (int filterR = 0; filterR < filterWidth; filterR++)
			for (int filterC = 0; filterC < filterWidth; filterC++)
			{
				int filterX_Val = dc_filterX[filterR*filterWidth + filterC];
				int filterY_Val = dc_filterY[filterR*filterWidth + filterC];

				int inPixelsR = r - 1 + filterR;
				int inPixelsC = c - 1 + filterC;
				inPixelsR = min(max(0, inPixelsR), height - 1);
				inPixelsC = min(max(0, inPixelsC), width - 1);
				
				int grayPixel = grayPixels[inPixelsR*width + inPixelsC];

				imp_X += filterX_Val * grayPixel;
				imp_Y += filterY_Val * grayPixel;
			}
		// printf("%d\n", imp_X);
		importance[r*width + c] = sqrt(float(imp_X*imp_X + imp_Y*imp_Y)); 
	}
}

__global__ void calculateImportanceKernel3(uint8_t * grayPixels, 
		int width, int height, int filterWidth, 
		int * importance)
{
// TODO
	int c= blockIdx.x * blockDim.x + threadIdx.x;
	int r= blockIdx.y * blockDim.y + threadIdx.y;

	if (r < height && c < width)
	{
		// Loads data to SMEM
		extern __shared__ uint8_t s_grayPixels[];
		
		int s_row= blockDim.y + filterWidth/2*2,
			cur_r= threadIdx.y + filterWidth/2,
			cur_c= threadIdx.x + filterWidth/2;

		s_grayPixels[cur_r*s_row + cur_c]= grayPixels[r * width + c];

		// Calculate edge value
		int top_blk= !threadIdx.y * filterWidth/2,
			bottom_blk= (threadIdx.y== blockDim.y - 1) * filterWidth/2, 
			left_blk= !threadIdx.x * filterWidth/2, 
			right_blk= (threadIdx.x== blockDim.x - 1 ) * filterWidth/2;

		for (int i= -top_blk; i <= bottom_blk; i++) //row
			for (int j= -left_blk; j <= right_blk; j++) //col
				s_grayPixels[(cur_r + i)*s_row + (cur_c + j)]= grayPixels[min(max(0, r + i), height - 1)*width + min(max(0, c + j), width - 1)];

		__syncthreads();
	
		// Do with data in SMEM
		int imp_X = 0, imp_Y = 0;
		for (int filterR = 0; filterR < filterWidth; filterR++)
			for (int filterC = 0; filterC < filterWidth; filterC++)
			{
				int filterX_Val = dc_filterX[filterR*filterWidth + filterC];
				int filterY_Val = dc_filterY[filterR*filterWidth + filterC];

				int s_R = cur_r - filterWidth/2 + filterR;
				int s_C = cur_c - filterWidth/2 + filterC;
				int grayPixel = s_grayPixels[s_R*s_row + s_C];

				imp_X += filterX_Val * grayPixel;
				imp_Y += filterY_Val * grayPixel;
			}

		importance[r*width + c] = sqrt(float(imp_X*imp_X + imp_Y*imp_Y)); 
	}
}

// Calculate importance kernel
__global__ void calculateImportanceKernel1(uint8_t * grayPixels, 
 		int * filterX, int * filterY,
		int width, int height, int filterWidth, 
		int * importance)
{
// TODO
	int c= blockIdx.x * blockDim.x + threadIdx.x;
	int r= blockIdx.y * blockDim.y + threadIdx.y;

	if (r < height && c < width)
	{
		int imp_X = 0, imp_Y = 0;
		for (int filterR = 0; filterR < filterWidth; filterR++)
			for (int filterC = 0; filterC < filterWidth; filterC++)
			{
				int filterX_Val = filterX[filterR*filterWidth + filterC];
				int filterY_Val = filterY[filterR*filterWidth + filterC];

				int inPixelsR = r - 1 + filterR;
				int inPixelsC = c - 1 + filterC;
				inPixelsR = min(max(0, inPixelsR), height - 1);
				inPixelsC = min(max(0, inPixelsC), width - 1);
				
				int grayPixel = grayPixels[inPixelsR*width + inPixelsC];

				imp_X += filterX_Val * grayPixel;
				imp_Y += filterY_Val * grayPixel;
			}
		// printf("%d\n", imp_X);
		importance[r*width + c] = sqrt(float(imp_X*imp_X + imp_Y*imp_Y)); 
	}
}


void seamCarvingbyDevice(uchar3 * inPixels, int width, int height,
		uint8_t * outGrayPixels, uchar3 * outPixels, int * outImportance,
		int * filterX, int * filterY, 
		int pixelsReduce, int kernel, dim3 blockSize=dim3(1)
		)
{
	GpuTimer timer;
	timer.Start();
	printf("\n\nUsing kernel %d... \n", kernel);
// Allocate device memories
	uchar3 * d_in;
	uint8_t * d_gray;
	int * d_importance, * d_filterX, * d_filterY;
	size_t imageSize = width * height, filterWidth = 3;

	// Set grid size and call kernel (remember to check kernel error)
	dim3 gridSize((width-1)/blockSize.x + 1, (height-1)/blockSize.y + 1);
	size_t shared_PixelSize = (blockSize.x + 2) * (blockSize.y + 2) * sizeof(int);

	CHECK(cudaMalloc(&d_in, imageSize * sizeof(uchar3)));
	CHECK(cudaMalloc(&d_gray, imageSize * sizeof(uint8_t)));
	CHECK(cudaMalloc(&d_importance, imageSize * sizeof(int)));
	if (kernel == 1) 
	{
		CHECK(cudaMalloc(&d_filterX, 9 * sizeof(int)));
		CHECK(cudaMalloc(&d_filterY, 9 * sizeof(int)));
	}
// Copy data to device memories
		// copy to, copy from, num of bytes, type
	CHECK(cudaMemcpy(d_in, inPixels, imageSize * sizeof(uchar3), cudaMemcpyHostToDevice));
	if (kernel == 1) 
	{
		CHECK(cudaMemcpy(d_filterX, filterX, 9 * sizeof(uchar3), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_filterY, filterY, 9 * sizeof(uchar3), cudaMemcpyHostToDevice));
	}
		// copy data from "filter" (on host) to "dc_filter" (on CMEM of device)
	CHECK(cudaMemcpyToSymbol(dc_filterX, filterX, 9 * sizeof(int)));
	CHECK(cudaMemcpyToSymbol(dc_filterY, filterY, 9 * sizeof(int)));


//TODO-> Gray Scale kernel
	uint8_t * grayPixels= (uint8_t *)malloc(width * height * sizeof(uint8_t));

	convertRgb2GrayKernel<<<gridSize, blockSize>>>(d_in, width, height, d_gray);
		// Check kernel errors
	cudaDeviceSynchronize();
	CHECK(cudaGetLastError());
	CHECK(cudaMemcpy(outGrayPixels, d_gray, imageSize * sizeof(uint8_t), cudaMemcpyDeviceToHost)); // for comparison
	CHECK(cudaMemcpy(grayPixels, d_gray, imageSize * sizeof(uint8_t), cudaMemcpyDeviceToHost));


// input reduce pixels 
	//using for loop ... 

	
	for (int loop = 0; loop < pixelsReduce; loop++)
	{
		dim3 grid((width-1)/blockSize.x + 1, (height-1)/blockSize.y + 1);
		//TODO-> Importance kernel
		int * importance = (int *)malloc(width * height * sizeof(int));
		if (kernel == 1)
			calculateImportanceKernel1<<<grid, blockSize>>>(d_gray, d_filterX, d_filterY, width, height, filterWidth, d_importance);
		else if (kernel == 2)
			calculateImportanceKernel2<<<grid, blockSize>>>(d_gray, width, height, filterWidth, d_importance);
		else if (kernel == 3)
			calculateImportanceKernel3<<<grid, blockSize, shared_PixelSize>>>(d_gray, width, height, filterWidth, d_importance);
			

		cudaDeviceSynchronize();
		CHECK(cudaGetLastError());
		CHECK(cudaMemcpy(importance, d_importance, width * height * sizeof(int), cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(outImportance, d_importance, width * height * sizeof(int), cudaMemcpyDeviceToHost)); // for comparison
		
		//TODO-> Direction
		calculateDirection(importance, width, height);
		
		//TODO-> Remove
		seamCarving(importance, width, height, loop, grayPixels, inPixels);
		CHECK(cudaMemcpy(d_gray, grayPixels, (width - 1) * height * sizeof(uint8_t), cudaMemcpyHostToDevice));

		// Reduce width every loop
		width--;

		free(importance);
	}

	// Write output
	for (int R = 0; R < height; R++)
		for (int C = 0; C < width; C++)
		{
			int i = R*width + C;
			outPixels[i] = inPixels[i];
		}

	free(grayPixels);

	

	// TODO: Free device memories
	CHECK(cudaFree(d_importance));
	CHECK(cudaFree(d_in));
	CHECK(cudaFree(d_gray));
	if (kernel == 1) 
	{
		CHECK(cudaFree(d_filterX));
		CHECK(cudaFree(d_filterY));
	}
	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time (use device): %f ms\n\n", time);
}

float computeError(uint8_t * a1, uint8_t * a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)
		err += (int)a1[i] != (int)a2[i];
	err /= n;
	return err;
}

float computeError(int * a1, int * a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)
		err += a1[i] != a2[i];
	err /= n;
	return err;
}

float computeError(uchar3 * a1, uchar3 * a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)
	{
		err += (int)a1[i].x != (int)a2[i].x||(int)a1[i].y != (int)a2[i].y||(int)a1[i].z != (int)a2[i].z;
	}
	err /= n;
	return err;
}

void printError(uchar3 * correctOutPixels, uchar3 * outPixels,
		uint8_t * correctGrayPixels, uint8_t * grayPixels,
		int * correctImportance, int * importance,
		int width, int height
		)
{
	printf("Gray Scale error: %f\n", computeError(correctGrayPixels, grayPixels, width * height));

	printf("Importance error: %f\n", computeError(correctImportance, importance, width * height));

	printf("Seam Carving error: %f\n", computeError(correctOutPixels, outPixels, width * height));
}

int main(int argc, char ** argv)
{
	if (argc != 4 && argc != 6)
	{
		printf("The number of arguments is invalid\n");
		return 0;
	}
	// printDeviceInfo();

// Read input image file
	int width, height;
	uchar3 * inPixels;
	readPnm(argv[1], width, height, inPixels);
	printf("\nImage size (width x height): %i x %i\n", width, height);

// Set up filter 
	// int filterWidth = 9;
	int filter_x_sobel[9]= {1, 0, -1, 
													2, 0, -2, 
													1, 0, -1};

	int filter_y_sobel[9]= {1, 2, 1, 
													0, 0, 0, 
													-1, -2, -1};

// Host ---------

	// TODO
	int pixelsReduce = atoi(argv[3]);
	if (pixelsReduce >= width)
		return 0;

	uint8_t * correctGrayPixels = (uint8_t *)malloc(width * height * sizeof(uint8_t));
	uchar3 * correctOutPixels = (uchar3 *)malloc((width - pixelsReduce) * height * sizeof(uchar3));
	int * correctImportance = (int *)malloc(width * height * sizeof(int));
	seamCarvingbyHost(inPixels, width, height, correctGrayPixels, correctOutPixels, correctImportance, filter_x_sobel, filter_y_sobel, pixelsReduce);


// Device -------
  dim3 blockSize(32, 32); // Default
		if (argc == 6)
	{
		blockSize.x = atoi(argv[4]);
		blockSize.y = atoi(argv[5]);
	}

	// using kernel 1 -----
	readPnm(argv[1], width, height, inPixels);
	printf("\nImage size (width x height): %i x %i\n", width, height);

	uint8_t * grayPixels1 = (uint8_t *)malloc(width * height * sizeof(uint8_t));
	uchar3 * outPixels1 = (uchar3 *)malloc((width - pixelsReduce) * height * sizeof(uchar3));
	int * importance1 = (int *)malloc(width * height * sizeof(int));
	
	seamCarvingbyDevice(inPixels, width, height, grayPixels1, outPixels1, importance1, filter_x_sobel, filter_y_sobel, pixelsReduce, 1, blockSize);

	printError(correctOutPixels, outPixels1, 
					correctGrayPixels, grayPixels1, 
					correctImportance, importance1,
					width, height);	

	// using kernel 2 ------
	readPnm(argv[1], width, height, inPixels);
	printf("\nImage size (width x height): %i x %i\n", width, height);

	uint8_t * grayPixels2 = (uint8_t *)malloc(width * height * sizeof(uint8_t));
	uchar3 * outPixels2 = (uchar3 *)malloc((width - pixelsReduce) * height * sizeof(uchar3));
	int * importance2 = (int *)malloc(width * height * sizeof(int));


	seamCarvingbyDevice(inPixels, width, height, grayPixels2, outPixels2, importance2, filter_x_sobel, filter_y_sobel, pixelsReduce, 2, blockSize);

	printError(correctOutPixels, outPixels2, 
					correctGrayPixels, grayPixels2, 
					correctImportance, importance2,
					width, height);


	// using kernel 3 ------
	readPnm(argv[1], width, height, inPixels);
	printf("\nImage size (width x height): %i x %i\n", width, height);

	uint8_t * grayPixels3 = (uint8_t *)malloc(width * height * sizeof(uint8_t));
	uchar3 * outPixels3 = (uchar3 *)malloc((width - pixelsReduce) * height * sizeof(uchar3));
	int * importance3 = (int *)malloc(width * height * sizeof(int));
	
	seamCarvingbyDevice(inPixels, width, height, grayPixels3, outPixels3, importance3, filter_x_sobel, filter_y_sobel, pixelsReduce, 3, blockSize);

	printError(correctOutPixels, outPixels3, 
					correctGrayPixels, grayPixels3, 
					correctImportance, importance3,
					width, height);

	printf("\nImage size after resize (width x height): %i x %i\n", width - pixelsReduce, height);

// Write results to files
  char * outFileNameBase = strtok(argv[2], "."); // Get rid of extension
	writePnm(correctOutPixels, width - pixelsReduce, height, concatStr(outFileNameBase, "_host.pnm"));

	writePnm(outPixels1, width - pixelsReduce, height, concatStr(outFileNameBase, "_device1.pnm"));

	writePnm(outPixels2, width - pixelsReduce, height, concatStr(outFileNameBase, "_device2.pnm"));

	writePnm(outPixels3, width - pixelsReduce, height, concatStr(outFileNameBase, "_device3.pnm"));

	// Free memories
	free(inPixels);
	free(correctOutPixels);
	free(correctGrayPixels);
	free(correctImportance);

	free(grayPixels1);
	free(outPixels1);
	free(importance1);

	free(grayPixels2);
	free(outPixels2);
	free(importance2);

	free(grayPixels3);
	free(outPixels3);
	free(importance3);
	return 0;
}
