#define ESCAPE_RADIUS_SQ 2000.0*2000.0	//the escape radius

//enum for transfer function types
enum TRANSFER_FUNCTION{
	ESCAPE_VELOCITY,
	HISTOGRAM_ESCAPE_VELOCITY,
	HISTOGRAM_NORMALISED_ITERATION_COUNT,
	RANDOM_NORMALISED_ITERATION_COUNT
};
typedef enum TRANSFER_FUNCTION TRANSFER_FUNCTION;

enum HISTOGRAM_METHOD{
	OMP_CRITICAL,
	OMP_ATOMIC,
	OMP_MASTER,
	OMP_SERIAL
};
typedef enum HISTOGRAM_METHOD HISTOGRAM_METHOD;

//rgb structure
struct rgb{
	unsigned char r;
	unsigned char g;
	unsigned char b;
};
typedef struct rgb rgb;

rgb ev_transfer(int x, int y);				//escape velocity transfer
rgb h_ev_tranfer(int x, int y);				//histogram escape velocity transfer with equally distributed colours regardless of iterations
rgb h_nic_transfer(int x, int y);			//histogram normalised iteration count (NIC) transfer with smooth shading (no banding) and equally distributed colours regardless of iterations
rgb rand_nic_transfer(int x, int y);		//random colours normalised iteration count (NIC) transfer with smooth shading (no banding)
