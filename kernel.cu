#define _USE_MATH_DEFINES

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

static uint8_t* buffer = 0;
static double* vecl = 0;
static int** gsyms = 0;
static int** ggstruct = 0;
static int** gperm = 0;
static int** gsign = 0;
static int** topo = 0;

static int* astnum = 0;
static int* lidx = 0;
static int* astnumplus = 0;
static double* astrad = 0;
static int* asttopo = 0;
static double* astpos0 = 0;
static double* astpos1 = 0;
static double* astpos2 = 0;
static int* astplusidx = 0;
static int* astsym = 0;
static double** astmat = 0;
static double** astmatv = 0;

static double* astv0 = 0;
static double* astv1 = 0;
static double* astv2 = 0;

static int* cnumplus = 0;
static double* ccpos0 = 0;
static double* ccpos1 = 0;
static double* ccpos2 = 0;
static int* ccsym = 0;

static int* lnumplus = 0;
static double* lpos0 = 0;
static double* lpos1 = 0;
static double* lpos2 = 0;
static double* lv0 = 0;
static double* lv1 = 0;
static double* lv2 = 0;

static double* seedist = 0;
static double* dist = 0;

static bool* lout = 0;
static int* loutf = 0;

static int* gginv = 0;

static bool* rip = 0;

static double* haz = 0;


__device__ void matmult(double* m1, double* m2)
{
	double m3[9]{};
	m3[0] = m1[0] * m2[0] + m1[1] * m2[3] + m1[2] * m2[6];
	m3[1] = m1[0] * m2[1] + m1[1] * m2[4] + m1[2] * m2[7];
	m3[2] = m1[0] * m2[2] + m1[1] * m2[5] + m1[2] * m2[8];
	m3[3] = m1[3] * m2[0] + m1[4] * m2[3] + m1[5] * m2[6];
	m3[4] = m1[3] * m2[1] + m1[4] * m2[4] + m1[5] * m2[7];
	m3[5] = m1[3] * m2[2] + m1[4] * m2[5] + m1[5] * m2[8];
	m3[6] = m1[6] * m2[0] + m1[7] * m2[3] + m1[8] * m2[6];
	m3[7] = m1[6] * m2[1] + m1[7] * m2[4] + m1[8] * m2[7];
	m3[8] = m1[6] * m2[2] + m1[7] * m2[5] + m1[8] * m2[8];

	for(int i=0;i<9;i++) m2[i] = m3[i];
}

__device__ int rnbw(int nbframe)
{
	int r = 0, g = 0, b = 0;
	double tmp;
	double x;

	x = fmod(nbframe * 0.006, 1.0);
	tmp = fmod(x, 1.0 / 6.0);

	if (x < 1.0 / 6.0)
	{
		r = 255;
		g = 1530 * tmp;
	}
	else if (x < 1.0 / 3.0)
	{
		g = 255;
		r = 255 - 1530 * tmp;
	}
	else if (x < 0.5)
	{
		g = 255;
		b = 1530 * tmp;
	}
	else if (x < 2.0 / 3.0)
	{
		b = 255;
		g = 255 - 1530 * tmp;
	}
	else if (x < 5.0 / 6.0)
	{
		b = 255;
		r = 1530 * tmp;
	}
	else
	{
		r = 255;
		b = 255 - 1530 * tmp;
	}

	return r + 256 * g + 256 * 256 * b;
}

__device__ int rnbw2(double x)
{
	int r = 0, g = 0, b = 0;
	double tmp;

	tmp = fmod(x, 1.0 / 6.0);

	if (x < 1.0 / 6.0)
	{
		r = 255;
		g = 1530 * tmp;
	}
	else if (x < 1.0 / 3.0)
	{
		g = 255;
		r = 255 - 1530 * tmp;
	}
	else if (x < 0.5)
	{
		g = 255;
		b = 1530 * tmp;
	}
	else if (x < 2.0 / 3.0)
	{
		b = 255;
		g = 255 - 1530 * tmp;
	}
	else if (x < 5.0 / 6.0)
	{
		b = 255;
		r = 1530 * tmp;
	}
	else
	{
		r = 255;
		b = 255 - 1530 * tmp;
	}

	return r + 256 * g + 256 * 256 * b;
}

__global__ void setast(uint8_t* buffer, double* astrad, double* astpos0, double* astpos1, double* astpos2, double* astv0, double* astv1, double* astv2, int* astnum, double* seedist, double* dist, double** astmat, double** astmatv)
{
	int i,l;
	int rand = 1;
	double maxang = 0.005;
	double tmp;
	double tmpmat[9]{};


	seedist[0] = 5;
	dist[0] = 2;
	astnum[0] = 1;

	astrad[0] = 0.2;
	astpos0[0] = 0;
	astpos1[0] = 0;
	astpos2[0] = 0;
	astv0[0] =0.001;
	astv1[0] = 0;
	astv2[0] =0;

	for (i = 0; i < 32; i++)
	{
		astmat[i][0] = 1;
		astmat[i][1] = 0;
		astmat[i][2] = 0;
		astmat[i][3] = 0;
		astmat[i][4] = 1;
		astmat[i][5] = 0;
		astmat[i][6] = 0;
		astmat[i][7] = 0;
		astmat[i][8] = 1;

		astmatv[i][0] = 1;
		astmatv[i][1] = 0;
		astmatv[i][2] = 0;
		astmatv[i][3] = 0;
		astmatv[i][4] = 1;
		astmatv[i][5] = 0;
		astmatv[i][6] = 0;
		astmatv[i][7] = 0;
		astmatv[i][8] = 1;

		for (l = 0; l < 10; l++) rand = (60493 * rand + 11) % 479001599;

		tmp = (2.0 * maxang) * (rand / 479001598.0) - maxang;

		tmpmat[0] = 1;
		tmpmat[1] = 0;
		tmpmat[2] = 0;
		tmpmat[3] = 0;
		tmpmat[4] = cos(tmp);
		tmpmat[5] = -sin(tmp);
		tmpmat[6] = 0;
		tmpmat[7] = sin(tmp);
		tmpmat[8] = cos(tmp);

		matmult(tmpmat, astmatv[i]);

		for (l = 0; l < 10; l++) rand = (60493 * rand + 11) % 479001599;

		tmp = (2.0 * maxang) * (rand / 479001598.0) - maxang;

		tmpmat[0] = cos(tmp);
		tmpmat[1] = 0;
		tmpmat[2] = sin(tmp);
		tmpmat[3] = 0;
		tmpmat[4] = 1;
		tmpmat[5] = 0;
		tmpmat[6] = -sin(tmp);
		tmpmat[7] = 0;
		tmpmat[8] = cos(tmp);

		matmult(tmpmat, astmatv[i]);

		for (l = 0; l < 10; l++) rand = (60493 * rand + 11) % 479001599;

		tmp = (2.0 * maxang) * (rand / 479001598.0) - maxang;

		tmpmat[0] = cos(tmp);
		tmpmat[1] = -sin(tmp);
		tmpmat[2] = 0;
		tmpmat[3] = sin(tmp);
		tmpmat[4] = cos(tmp);
		tmpmat[5] = 0;
		tmpmat[6] = 0;
		tmpmat[7] = 0;
		tmpmat[8] = 1;

		matmult(tmpmat, astmatv[i]);

	}
	
	for (i = 0; i < 4 * 1920 * 1080; i++) buffer[i] = 255;
	
}

__global__ void rotast(double** astmat, double** astmatv)
{
	int tmp = blockIdx.x * blockDim.x + threadIdx.x;

	matmult(astmatv[tmp], astmat[tmp]);
}

__global__ void modfovp(double* dist, bool fovp, bool fovm, bool resetflag)
{
	if (resetflag) dist[0] = 2.0;
	else
	{
		if (fovp) dist[0] += 0.05;
		else
		{
			if (dist[0] > 0.05) dist[0] -= 0.05;
		}
	}
}

__global__ void modfov(double* dist, double* vecl)
{
	int tmp = blockIdx.x * blockDim.x + threadIdx.x;
	int tmpx = tmp % 1920;
	int tmpy = (tmp - tmpx) / 1920;
		
		double sqsz = 0.01 / 6;

		double vec0, vec1, vec2;
		double addy0, addy1, addy2;
		double addz0, addz1, addz2;
		double vecn0, vecn1, vecn2;
		double x00 = 1, x01 = 0, x02 = 0;
		double x10 = 0, x11 = 1, x12 = 0;
		double x20 = 0, x21 = 0, x22 = 1;
		double multy = (1 - 1920) * sqsz / 2;
		double multz = (1080 - 1) * sqsz / 2;

		vec0 = dist[0] * x00 + multy * x10 + multz * x20;
		vec1 = dist[0] * x01 + multy * x11 + multz * x21;
		vec2 = dist[0] * x02 + multy * x12 + multz * x22;

		addy0 = sqsz * x10;
		addy1 = sqsz * x11;
		addy2 = sqsz * x12;

		addz0 = -sqsz * x20;
		addz1 = -sqsz * x21;
		addz2 = -sqsz * x22;


			vecn0 = vec0 + tmpx * addy0 + tmpy * addz0;
			vecn1 = vec1 + tmpx * addy1 + tmpy * addz1;
			vecn2 = vec2 + tmpx * addy2 + tmpy * addz2;

			vecl[tmp] = 1.0 / sqrt(vecn0 * vecn0 + vecn1 * vecn1 + vecn2 * vecn2);
		
}

__global__ void modast(double* astrad, double* astpos0, double* astpos1, double* astpos2, double* astv0, double* astv1, double* astv2, int** topo, int** gperm, int** gsign, int currenttopo, int* astnum, int* astnumplus, double* lpos0, double* lpos1, double* lpos2, double* lv0, double* lv1, double* lv2, int* lnumplus, double pos0, double pos1, double pos2, double* ccpos0, double* ccpos1, double* ccpos2, int* cnumplus, bool fireflag, double v0, double v1, double v2, bool* lout, int* loutf, int nbframe, int* gginv, int* astplusidx, bool* rip, double* haz, int* lidx,bool resetflag, double* seedist, bool sdp, bool sdm, double cv0, double cv1, double cv2, bool w0, bool w1, bool w2, bool w3, bool w4, bool w5, int* ccsym, int* astsym, double** astmat, double** astmatv, int* asttopo)
{
	double lsize = 0.01;
	
	int l;
	int j,ii;
	int newsym;
	int ppos0, ppos1, ppos2;
	double postmp[3]{};
	double nastv[3]{};
	double nnastv[3]{};
	double disto;
	double theta, phi;
	double coord0, coord1, coord2;
	double dpos0, dpos1, dpos2;
	double tmp0, tmp1, tmp2;
	double csize = 0.05;
	int i, k;
	int astnumplustmp = 0;
	int cnumplustmp = 1;
	int lnumplustmp = 0;
	int tmpidx;
	double hazmax = 0;
	double tmpd;
	int tmpproj, projpp1, projpp2;
	double tmpmat;

	bool wall[6]{};

	wall[0] = w0;
	wall[1] = w1;
	wall[2] = w2;
	wall[3] = w3;
	wall[4] = w4;
	wall[5] = w5;

	if (sdp) seedist[0] += 0.1;

	if (sdm)
	{
		if (seedist[0] > 0.1) seedist[0] -= 0.1;
	}

	
	if (resetflag)
	{
		rip[0] = false;

		astnum[0] = 1;
		seedist[0] = 5;
		astrad[0] = 0.2;
		astpos0[0] = 0;
		astpos1[0] = 0;
		astpos2[0] = 0;
		astv0[0] = 0.001;
		astv1[0] = 0;
		astv2[0] = 0;

		lout[0] = false;
		lout[1] = false;
		lout[2] = false;
	}
	
	if (fireflag)
	{
		if (!lout[0])
		{
			lout[0] = true;
			loutf[0] = nbframe;

			lv0[0] = cv0+v0 / 50;
			lv1[0] =cv1+ v1 / 50;
			lv2[0] = cv2+ v2 / 50;

			lpos0[0] = pos0 - 0.5  -lv0[0];
			lpos1[0] = pos1 - 0.5 -lv1[0];
			lpos2[0] = pos2 - 0.5 -lv2[0];
		}
		else if (!lout[1])
		{
			lout[1] = true;
			loutf[1] = nbframe;

			lv0[1] = cv0+ v0 / 50;
			lv1[1] = cv1+ v1 / 50;
			lv2[1] = cv2+ v2 / 50;

			lpos0[1] = pos0 - 0.5  - lv0[1];
			lpos1[1] = pos1 - 0.5 - lv1[1];
			lpos2[1] = pos2 - 0.5  - lv2[1];
		}
		else if (!lout[2])
		{
			lout[2] = true;
			loutf[2] = nbframe;

			lv0[2] = cv0+ v0 / 50;
			lv1[2] = cv1+ v1 / 50;
			lv2[2] = cv2+ v2 / 50;

			lpos0[2] = pos0 - 0.5  - lv0[2];
			lpos1[2] = pos1 - 0.5  - lv1[2];
			lpos2[2] = pos2 - 0.5  - lv2[2];
		}
	}

	for (i = 0; i < 3; i++) if (lout[i] && nbframe - loutf[i] > 200) lout[i] = false;
	

	for (l = 0; l < astnum[0]; l++)
	{

		astpos0[l] += astv0[l];
		astpos1[l] += astv1[l];
		astpos2[l] += astv2[l];

		if (wall[0])
		{
			tmpd = astpos2[l] + astrad[l] - 0.5;
			if (tmpd > 0)
			{
				astpos2[l] -= tmpd;
				astv2[l] *= -1;
			}
		}
		if (wall[1])
		{
			tmpd = astpos0[l] + astrad[l] - 0.5;
			if (tmpd > 0)
			{
				astpos0[l] -= tmpd;
				astv0[l] *= -1;
			}
		}
		if (wall[2])
		{
			tmpd = astpos1[l] + astrad[l] - 0.5;
			if (tmpd > 0)
			{
				astpos1[l] -= tmpd;
				astv1[l] *= -1;
			}
		}
		if (wall[5])
		{
			tmpd = astpos2[l] - astrad[l] + 0.5;
			if (tmpd < 0)
			{
				astpos2[l] -= tmpd;
				astv2[l] *= -1;
			}
		}
		if (wall[3])
		{
			tmpd = astpos0[l] - astrad[l] + 0.5;
			if (tmpd < 0)
			{
				astpos0[l] -= tmpd;
				astv0[l] *= -1;
			}
		}
		if (wall[4])
		{
			tmpd = astpos1[l] - astrad[l] + 0.5;
			if (tmpd < 0)
			{
				astpos1[l] -= tmpd;
				astv1[l] *= -1;
			}
		}

		dpos0 = astpos0[l] + 0.5;
		dpos1 = astpos1[l] + 0.5;
		dpos2 = astpos2[l] + 0.5;

		ppos0 = dpos0;
		ppos1 = dpos1;
		ppos2 = dpos2;

		if (dpos0 < 0) ppos0--;
		if (dpos1 < 0) ppos1--;
		if (dpos2 < 0) ppos2--;

		if (ppos0 != 0 || ppos1 != 0 || ppos2 != 0)
		{
			ppos0 %= 12;
			ppos1 %= 12;
			ppos2 %= 12;
			if (ppos0 < 0)ppos0 += 12;
			if (ppos1 < 0)ppos1 += 12;
			if (ppos2 < 0)ppos2 += 12;


			newsym = topo[currenttopo][ppos2 + 12 * ppos1 + 12 * 12 * ppos0];

			tmpproj = asttopo[l];
			projpp2 = tmpproj % 12;
			tmpproj -= projpp2;
			tmpproj /= 12;
			projpp1 = tmpproj % 12;
			tmpproj -= projpp1;
			tmpproj /= 12;
			
			ppos0 += tmpproj;
			ppos0 %= 12;
			ppos1 += projpp1;
			ppos1 %= 12;
			ppos2 += projpp2;
			ppos2 %= 12;

			asttopo[l] = ppos2 + 12 * ppos1 + 12 * 12 * ppos0;



			dpos0 = fmod(dpos0, 1.0);
			if (dpos0 < 0) dpos0++;
			dpos0 -= 0.5;
			dpos1 = fmod(dpos1, 1.0);
			if (dpos1 < 0) dpos1++;
			dpos1 -= 0.5;
			dpos2 = fmod(dpos2, 1.0);
			if (dpos2 < 0) dpos2++;
			dpos2 -= 0.5;

			postmp[0] = dpos0;
			postmp[1] = dpos1;
			postmp[2] = dpos2;

			dpos0 = gsign[newsym][0] * postmp[gperm[newsym][0]];
			dpos1 = gsign[newsym][1] * postmp[gperm[newsym][1]];
			dpos2 = gsign[newsym][2] * postmp[gperm[newsym][2]];



			astpos0[l] = dpos0;
			astpos1[l] = dpos1;
			astpos2[l] = dpos2;


			nastv[0] = astv0[l];
			nastv[1] = astv1[l];
			nastv[2] = astv2[l];

			for (j = 0; j < 3; j++) nnastv[j] = gsign[newsym][j] * nastv[gperm[newsym][j]];

			astv0[l] = nnastv[0];
			astv1[l] = nnastv[1];
			astv2[l] = nnastv[2];
			
		}
	}
	
	for (i = 0; i < 3; i++)
	{
		if (lout[i])
		{

			lpos0[i] += lv0[i];
			lpos1[i] += lv1[i];
			lpos2[i] += lv2[i];

			if (wall[0])
			{
				tmpd = lpos2[i] + lsize - 0.5;
				if (tmpd > 0)
				{
					lpos2[i] -= tmpd;
					lv2[i] *= -1;
				}
			}
			if (wall[1])
			{
				tmpd = lpos0[i] + lsize - 0.5;
				if (tmpd > 0)
				{
					lpos0[i] -= tmpd;
					lv0[i] *= -1;
				}
			}
			if (wall[2])
			{
				tmpd = lpos1[i] + lsize - 0.5;
				if (tmpd > 0)
				{
					lpos1[i] -= tmpd;
					lv1[i] *= -1;
				}
			}
			if (wall[5])
			{
				tmpd = lpos2[i] - lsize + 0.5;
				if (tmpd < 0)
				{
					lpos2[i] -= tmpd;
					lv2[i] *= -1;
				}
			}
			if (wall[3])
			{
				tmpd = lpos0[i] - lsize + 0.5;
				if (tmpd < 0)
				{
					lpos0[i] -= tmpd;
					lv0[i] *= -1;
				}
			}
			if (wall[4])
			{
				tmpd = lpos1[i] - lsize + 0.5;
				if (tmpd < 0)
				{
					lpos1[i] -= tmpd;
					lv1[i] *= -1;
				}
			}


			dpos0 = lpos0[i] + 0.5;
			dpos1 = lpos1[i] + 0.5;
			dpos2 = lpos2[i] + 0.5;

			ppos0 = dpos0;
			ppos1 = dpos1;
			ppos2 = dpos2;

			if (dpos0 < 0) ppos0--;
			if (dpos1 < 0) ppos1--;
			if (dpos2 < 0) ppos2--;

			if (ppos0 != 0 || ppos1 != 0 || ppos2 != 0)
			{
				ppos0 %= 12;
				ppos1 %= 12;
				ppos2 %= 12;
				if (ppos0 < 0)ppos0 += 12;
				if (ppos1 < 0)ppos1 += 12;
				if (ppos2 < 0)ppos2 += 12;


				newsym = topo[currenttopo][ppos2 + 12 * ppos1 + 12 * 12 * ppos0];

				dpos0 = fmod(dpos0, 1.0);
				if (dpos0 < 0) dpos0++;
				dpos0 -= 0.5;
				dpos1 = fmod(dpos1, 1.0);
				if (dpos1 < 0) dpos1++;
				dpos1 -= 0.5;
				dpos2 = fmod(dpos2, 1.0);
				if (dpos2 < 0) dpos2++;
				dpos2 -= 0.5;

				postmp[0] = dpos0;
				postmp[1] = dpos1;
				postmp[2] = dpos2;

				dpos0 = gsign[newsym][0] * postmp[gperm[newsym][0]];
				dpos1 = gsign[newsym][1] * postmp[gperm[newsym][1]];
				dpos2 = gsign[newsym][2] * postmp[gperm[newsym][2]];



				lpos0[i] = dpos0;
				lpos1[i] = dpos1;
				lpos2[i] = dpos2;


				nastv[0] = lv0[i];
				nastv[1] = lv1[i];
				nastv[2] = lv2[i];

				for (j = 0; j < 3; j++) nnastv[j] = gsign[newsym][j] * nastv[gperm[newsym][j]];

				lv0[i] = nnastv[0];
				lv1[i] = nnastv[1];
				lv2[i] = nnastv[2];
			}
		}
	}


	for (l = 0; l < astnum[0]; l++)
	{
		for (j = l+1; j < astnum[0]; j++)
		{
			coord0 = astpos0[l] - astpos0[j];
			coord1 = astpos1[l] - astpos1[j];
			coord2 = astpos2[l] - astpos2[j];
			disto = sqrt(coord0 * coord0 + coord1 * coord1 + coord2 * coord2);
			if (disto < astrad[l] + astrad[j])
			{
				astv0[l] = coord0 * .001 / disto;
				astv1[l] = coord1 * .001 / disto;
				astv2[l] = coord2 * .001 / disto;

				astv0[j] = -coord0 * .001 / disto;
				astv1[j] = -coord1 * .001 / disto;
				astv2[j] = -coord2 * .001 / disto;
			}
		}
		for (j = astnum[0]; j < astnum[0] + astnumplus[0]; j++)
		{
			if (astplusidx[j] != l)
			{
				coord0 = astpos0[l] - astpos0[j];
				coord1 = astpos1[l] - astpos1[j];
				coord2 = astpos2[l] - astpos2[j];
				disto = sqrt(coord0 * coord0 + coord1 * coord1 + coord2 * coord2);
				if (disto < astrad[l] + astrad[j])
				{
					astv0[l] = coord0 * .001 / disto;
					astv1[l] = coord1 * .001 / disto;
					astv2[l] = coord2 * .001 / disto;
				}
			}
		}
	}
	
	for (l = 0; l < astnum[0]; l++)
	{
		for (i = -1; i < 2; i++)
		{
			for (j = -1; j < 2; j++)
			{
				for (k = -1; k < 2; k++)
				{
					if (i != 0 || j != 0 || k != 0)
					{
						ppos0 = i;
						ppos1 = j;
						ppos2 = k;

						postmp[0] = astpos0[l];
						postmp[1] = astpos1[l];
						postmp[2] = astpos2[l];

						if (ppos0 < 0)ppos0 += 12;
						if (ppos1 < 0)ppos1 += 12;
						if (ppos2 < 0)ppos2 += 12;

						newsym = gginv[topo[currenttopo][ppos2 + 12 * ppos1 + 12 * 12 * ppos0]];

						dpos0 = gsign[newsym][0] * postmp[gperm[newsym][0]];
						dpos1 = gsign[newsym][1] * postmp[gperm[newsym][1]];
						dpos2 = gsign[newsym][2] * postmp[gperm[newsym][2]];

						dpos0 += i;
						dpos1 += j;
						dpos2 += k;

						if ((dpos0 - astrad[l] < 0.5 && dpos0 - astrad[l]> -0.5) || (dpos0 + astrad[l] < 0.5 && dpos0 + astrad[l]> -0.5))
						{
							if ((dpos1 - astrad[l] < 0.5 && dpos1 - astrad[l]> -0.5) || (dpos1 + astrad[l] < 0.5 && dpos1 + astrad[l]> -0.5))
							{
								if ((dpos2 - astrad[l] < 0.5 && dpos2 - astrad[l]> -0.5) || (dpos2 + astrad[l] < 0.5 && dpos2 + astrad[l]> -0.5))
								{
									astrad[astnum[0] + astnumplustmp] = astrad[l];
									astpos0[astnum[0] + astnumplustmp] = dpos0;
									astpos1[astnum[0] + astnumplustmp] = dpos1;
									astpos2[astnum[0] + astnumplustmp] = dpos2;
									astplusidx[astnum[0] + astnumplustmp] = l;
									astsym[astnum[0] + astnumplustmp] = gginv[newsym];
									astnumplustmp++;
								}
							}
						}



					}
				}
			}
		}
	}
	
	ccpos0[0] = pos0 - 0.5;
	ccpos1[0] = pos1 - 0.5;
	ccpos2[0] = pos2 - 0.5;

	for (i = -1; i < 2; i++)
	{
		for (j = -1; j < 2; j++)
		{
			for (k = -1; k < 2; k++)
			{
				if (i != 0 || j != 0 || k != 0)
				{
					ppos0 = i;
					ppos1 = j;
					ppos2 = k;

					postmp[0] = ccpos0[0];
					postmp[1] = ccpos1[0];
					postmp[2] = ccpos2[0];

					if (ppos0 < 0)ppos0 += 12;
					if (ppos1 < 0)ppos1 += 12;
					if (ppos2 < 0)ppos2 += 12;

					newsym = gginv[topo[currenttopo][ppos2 + 12 * ppos1 + 12*12*ppos0]];

				

					dpos0 = gsign[newsym][0] * postmp[gperm[newsym][0]];
					dpos1 = gsign[newsym][1] * postmp[gperm[newsym][1]];
					dpos2 = gsign[newsym][2] * postmp[gperm[newsym][2]];

					dpos0 += i;
					dpos1 += j;
					dpos2 += k;

					if ((dpos0 - csize < 0.5 && dpos0 - csize> -0.5) || (dpos0 + csize < 0.5 && dpos0 + csize> -0.5))
					{
						if ((dpos1 - csize < 0.5 && dpos1 - csize> -0.5) || (dpos1 + csize < 0.5 && dpos1 + csize> -0.5))
						{
							if ((dpos2 - csize < 0.5 && dpos2 - csize> -0.5) || (dpos2 + csize < 0.5 && dpos2 + csize> -0.5))
							{
								ccpos0[cnumplustmp] = dpos0;
								ccpos1[cnumplustmp] = dpos1;
								ccpos2[cnumplustmp] = dpos2;
								ccsym[cnumplustmp] = gginv[newsym];
								cnumplustmp++;
							}
						}
					}



				}
			}
		}
	}

	for (ii = 0; ii < 3; ii++)
	{
		if (lout[ii])
		{
			for (l = 0; l < astnum[0] + astnumplus[0]; l++)
			{
				coord0 = astpos0[l] - lpos0[ii];
				coord1 = astpos1[l] - lpos1[ii];
				coord2 = astpos2[l] - lpos2[ii];
				disto = sqrt(coord0 * coord0 + coord1 * coord1 + coord2 * coord2);
				if (disto < astrad[l] + lsize)
				{
					if (l >= astnum[0]) tmpidx = astplusidx[l];
					else tmpidx = l;

					if (astrad[tmpidx] < 2 * lsize)
					{
						astrad[tmpidx] = astrad[astnum[0] - 1];
						astpos0[tmpidx] = astpos0[astnum[0] - 1];
						astpos1[tmpidx] = astpos1[astnum[0] - 1];
						astpos2[tmpidx] = astpos2[astnum[0] - 1];
						astv0[tmpidx] = astv0[astnum[0] - 1];
						astv1[tmpidx] = astv1[astnum[0] - 1];
						astv2[tmpidx] = astv2[astnum[0] - 1];
						asttopo[tmpidx] = asttopo[astnum[0] - 1];
						
						for (k = 0; k < 9; k++)
						{
							tmpmat = astmat[tmpidx][k];
							astmat[tmpidx][k] = astmat[astnum[0]-1][k];
							astmat[astnum[0] - 1][k] = tmpmat;

							tmpmat = astmatv[tmpidx][k];
							astmatv[tmpidx][k] = astmatv[astnum[0] - 1][k];
							astmatv[astnum[0] - 1][k] = tmpmat;
						}
						
						
						astnum[0]--;

						lout[ii] = false;
				

						l = 2000;
					}
					else
					{
						theta = fmod((double)nbframe, 2 * M_PI);
						phi = fmod((double)(nbframe * nbframe), M_PI);

						tmp0 = sin(phi) * cos(theta);
						tmp1 = sin(phi) * sin(theta);
						tmp2 = cos(phi);

						astrad[tmpidx] /= 2;
						astrad[astnum[0]] = astrad[tmpidx];

						astpos0[astnum[0]] = astpos0[tmpidx] + tmp0 * astrad[tmpidx];
						astpos1[astnum[0]] = astpos1[tmpidx] + tmp1 * astrad[tmpidx];
						astpos2[astnum[0]] = astpos2[tmpidx] + tmp2 * astrad[tmpidx];

						astpos0[tmpidx] -= tmp0 * astrad[tmpidx];
						astpos1[tmpidx] -= tmp1 * astrad[tmpidx];
						astpos2[tmpidx] -= tmp2 * astrad[tmpidx];


						astv0[tmpidx] = -0.001 * tmp0;
						astv1[tmpidx] = -0.001 * tmp1;
						astv2[tmpidx] = -0.001 * tmp2;

						astv0[astnum[0]] = 0.001 * tmp0;
						astv1[astnum[0]] = 0.001 * tmp1;
						astv2[astnum[0]] = 0.001 * tmp2;

						asttopo[tmpidx] = 0;
						asttopo[astnum[0]] = 0;

						dpos0 = astpos0[tmpidx] + 0.5;
						dpos1 = astpos1[tmpidx] + 0.5;
						dpos2 = astpos2[tmpidx] + 0.5;

						ppos0 = dpos0;
						ppos1 = dpos1;
						ppos2 = dpos2;

						if (dpos0 < 0) ppos0--;
						if (dpos1 < 0) ppos1--;
						if (dpos2 < 0) ppos2--;

						if (ppos0 != 0 || ppos1 != 0 || ppos2 != 0)
						{
							ppos0 %= 12;
							ppos1 %= 12;
							ppos2 %= 12;
							if (ppos0 < 0)ppos0 += 12;
							if (ppos1 < 0)ppos1 += 12;
							if (ppos2 < 0)ppos2 += 12;


							newsym = topo[currenttopo][ppos2 + 12 * ppos1 + 12 * 12 * ppos0];

							tmpproj = asttopo[tmpidx];
							projpp2 = tmpproj % 12;
							tmpproj -= projpp2;
							tmpproj /= 12;
							projpp1 = tmpproj % 12;
							tmpproj -= projpp1;
							tmpproj /= 12;

							ppos0 += tmpproj;
							ppos0 %= 12;
							ppos1 += projpp1;
							ppos1 %= 12;
							ppos2 += projpp2;
							ppos2 %= 12;

							asttopo[tmpidx] = ppos2 + 12 * ppos1 + 12 * 12 * ppos0;

							dpos0 = fmod(dpos0, 1.0);
							if (dpos0 < 0) dpos0++;
							dpos0 -= 0.5;
							dpos1 = fmod(dpos1, 1.0);
							if (dpos1 < 0) dpos1++;
							dpos1 -= 0.5;
							dpos2 = fmod(dpos2, 1.0);
							if (dpos2 < 0) dpos2++;
							dpos2 -= 0.5;

							postmp[0] = dpos0;
							postmp[1] = dpos1;
							postmp[2] = dpos2;

							dpos0 = gsign[newsym][0] * postmp[gperm[newsym][0]];
							dpos1 = gsign[newsym][1] * postmp[gperm[newsym][1]];
							dpos2 = gsign[newsym][2] * postmp[gperm[newsym][2]];



							astpos0[tmpidx] = dpos0;
							astpos1[tmpidx] = dpos1;
							astpos2[tmpidx] = dpos2;


							nastv[0] = astv0[tmpidx];
							nastv[1] = astv1[tmpidx];
							nastv[2] = astv2[tmpidx];

							for (j = 0; j < 3; j++) nnastv[j] = gsign[newsym][j] * nastv[gperm[newsym][j]];

							astv0[tmpidx] = nnastv[0];
							astv1[tmpidx] = nnastv[1];
							astv2[tmpidx] = nnastv[2];
						}

						dpos0 = astpos0[astnum[0]] + 0.5;
						dpos1 = astpos1[astnum[0]] + 0.5;
						dpos2 = astpos2[astnum[0]] + 0.5;

						ppos0 = dpos0;
						ppos1 = dpos1;
						ppos2 = dpos2;

						if (dpos0 < 0) ppos0--;
						if (dpos1 < 0) ppos1--;
						if (dpos2 < 0) ppos2--;

						if (ppos0 != 0 || ppos1 != 0 || ppos2 != 0)
						{
							ppos0 %= 12;
							ppos1 %= 12;
							ppos2 %= 12;
							if (ppos0 < 0)ppos0 += 12;
							if (ppos1 < 0)ppos1 += 12;
							if (ppos2 < 0)ppos2 += 12;


							newsym = topo[currenttopo][ppos2 + 12 * ppos1 + 12 * 12 * ppos0];

							tmpproj = asttopo[astnum[0]];
							projpp2 = tmpproj % 12;
							tmpproj -= projpp2;
							tmpproj /= 12;
							projpp1 = tmpproj % 12;
							tmpproj -= projpp1;
							tmpproj /= 12;

							ppos0 += tmpproj;
							ppos0 %= 12;
							ppos1 += projpp1;
							ppos1 %= 12;
							ppos2 += projpp2;
							ppos2 %= 12;

							asttopo[astnum[0]] = ppos2 + 12 * ppos1 + 12 * 12 * ppos0;

							dpos0 = fmod(dpos0, 1.0);
							if (dpos0 < 0) dpos0++;
							dpos0 -= 0.5;
							dpos1 = fmod(dpos1, 1.0);
							if (dpos1 < 0) dpos1++;
							dpos1 -= 0.5;
							dpos2 = fmod(dpos2, 1.0);
							if (dpos2 < 0) dpos2++;
							dpos2 -= 0.5;

							postmp[0] = dpos0;
							postmp[1] = dpos1;
							postmp[2] = dpos2;

							dpos0 = gsign[newsym][0] * postmp[gperm[newsym][0]];
							dpos1 = gsign[newsym][1] * postmp[gperm[newsym][1]];
							dpos2 = gsign[newsym][2] * postmp[gperm[newsym][2]];



							astpos0[astnum[0]] = dpos0;
							astpos1[astnum[0]] = dpos1;
							astpos2[astnum[0]] = dpos2;


							nastv[0] = astv0[astnum[0]];
							nastv[1] = astv1[astnum[0]];
							nastv[2] = astv2[astnum[0]];

							for (j = 0; j < 3; j++) nnastv[j] = gsign[newsym][j] * nastv[gperm[newsym][j]];

							astv0[astnum[0]] = nnastv[0];
							astv1[astnum[0]] = nnastv[1];
							astv2[astnum[0]] = nnastv[2];
						}
						astnum[0]++;

						astnumplustmp = 0;
						for (l = 0; l < astnum[0]; l++)
						{
							for (i = -1; i < 2; i++)
							{
								for (j = -1; j < 2; j++)
								{
									for (k = -1; k < 2; k++)
									{
										if (i != 0 || j != 0 || k != 0)
										{
											ppos0 = i;
											ppos1 = j;
											ppos2 = k;

											postmp[0] = astpos0[l];
											postmp[1] = astpos1[l];
											postmp[2] = astpos2[l];

											if (ppos0 < 0)ppos0 += 12;
											if (ppos1 < 0)ppos1 += 12;
											if (ppos2 < 0)ppos2 += 12;

											newsym = gginv[topo[currenttopo][ppos2 + 12 * ppos1 + 12 * 12 * ppos0]];

											dpos0 = gsign[newsym][0] * postmp[gperm[newsym][0]];
											dpos1 = gsign[newsym][1] * postmp[gperm[newsym][1]];
											dpos2 = gsign[newsym][2] * postmp[gperm[newsym][2]];

											dpos0 += i;
											dpos1 += j;
											dpos2 += k;

											if ((dpos0 - astrad[l] < 0.5 && dpos0 - astrad[l]> -0.5) || (dpos0 + astrad[l] < 0.5 && dpos0 + astrad[l]> -0.5))
											{
												if ((dpos1 - astrad[l] < 0.5 && dpos1 - astrad[l]> -0.5) || (dpos1 + astrad[l] < 0.5 && dpos1 + astrad[l]> -0.5))
												{
													if ((dpos2 - astrad[l] < 0.5 && dpos2 - astrad[l]> -0.5) || (dpos2 + astrad[l] < 0.5 && dpos2 + astrad[l]> -0.5))
													{
														astrad[astnum[0] + astnumplustmp] = astrad[l];
														astpos0[astnum[0] + astnumplustmp] = dpos0;
														astpos1[astnum[0] + astnumplustmp] = dpos1;
														astpos2[astnum[0] + astnumplustmp] = dpos2;
														astplusidx[astnum[0] + astnumplustmp] = l;
														astsym[astnum[0] + astnumplustmp] = gginv[newsym];
														astnumplustmp++;
													}
												}
											}



										}
									}
								}
							}
						}


						lout[ii] = false;


						l = 2000;
					}
				}
			}
		}
	}


	for (ii = 0; ii < 3; ii++)
	{
		if (lout[ii])
		{
			for (i = -1; i < 2; i++)
			{
				for (j = -1; j < 2; j++)
				{
					for (k = -1; k < 2; k++)
					{
						if (i != 0 || j != 0 || k != 0)
						{
							ppos0 = i;
							ppos1 = j;
							ppos2 = k;

							postmp[0] = lpos0[ii];
							postmp[1] = lpos1[ii];
							postmp[2] = lpos2[ii];

							if (ppos0 < 0)ppos0 += 12;
							if (ppos1 < 0)ppos1 += 12;
							if (ppos2 < 0)ppos2 += 12;

							newsym = gginv[topo[currenttopo][ppos2 + 12 * ppos1 + 12 * 12 * ppos0]];

							dpos0 = gsign[newsym][0] * postmp[gperm[newsym][0]];
							dpos1 = gsign[newsym][1] * postmp[gperm[newsym][1]];
							dpos2 = gsign[newsym][2] * postmp[gperm[newsym][2]];

							dpos0 += i;
							dpos1 += j;
							dpos2 += k;

							if ((dpos0 - lsize < 0.5 && dpos0 - lsize> -0.5) || (dpos0 + lsize < 0.5 && dpos0 + lsize> -0.5))
							{
								if ((dpos1 - lsize < 0.5 && dpos1 - lsize> -0.5) || (dpos1 + lsize < 0.5 && dpos1 + lsize> -0.5))
								{
									if ((dpos2 - lsize < 0.5 && dpos2 - lsize> -0.5) || (dpos2 + lsize < 0.5 && dpos2 + lsize> -0.5))
									{
										lpos0[3+lnumplustmp] = dpos0;
										lpos1[3+lnumplustmp] = dpos1;
										lpos2[3+lnumplustmp] = dpos2;
										lidx[3 + lnumplustmp] = ii;
										lnumplustmp++;
									}
								}
							}



						}
					}
				}
			}
		}
	}

	for (l = 0; l < astnum[0] + astnumplus[0]; l++)
	{
			coord0 = astpos0[l] - ccpos0[0];
			coord1 = astpos1[l] - ccpos1[0];
			coord2 = astpos2[l] - ccpos2[0];
			disto = sqrt(coord0 * coord0 + coord1 * coord1 + coord2 * coord2);
			if (disto < astrad[l] + csize)
			{
				rip[0] = true;
			}
			
			haz[0] = 1- (disto- astrad[l]-csize) / ( 2*csize);
			if (haz[0] < 0) haz[0] = 0;
			if (haz[0] > hazmax) hazmax = haz[0];
	}

	for (l = 0; l < 3; l++)
	{
		if (lout[l] && nbframe - loutf[l] > 8)
		{
			coord0 = lpos0[l] - ccpos0[0];
			coord1 = lpos1[l] - ccpos1[0];
			coord2 = lpos2[l] - ccpos2[0];
			disto= sqrt(coord0 * coord0 + coord1 * coord1 + coord2 * coord2);
			if (disto < lsize + csize)
			{
				rip[0] = true;
			}

			haz[0] = 1 - (disto - lsize - csize) / (2 * csize);
			if (haz[0] < 0) haz[0] = 0;
			if (haz[0] > hazmax) hazmax = haz[0];
		}
	}

	
	for (l = 0; l < lnumplus[0]; l++)
	{
		if (nbframe - loutf[lidx[3 + l]] > 8)
		{
			coord0 = lpos0[3 + l] - ccpos0[0];
			coord1 = lpos1[3 + l] - ccpos1[0];
			coord2 = lpos2[3 + l] - ccpos2[0];
			disto = sqrt(coord0 * coord0 + coord1 * coord1 + coord2 * coord2);
			if (disto < lsize + csize)
			{
				rip[0] = true;
			}

			haz[0] = 1 - (disto - lsize - csize) / (2 * csize);
			if (haz[0] < 0) haz[0] = 0;
			if (haz[0] > hazmax) hazmax = haz[0];
		}
	}

	for (l = 0; l < astnum[0]; l++) { astplusidx[l] = l; astsym[l] = 0; }

	haz[0] = hazmax;
	lnumplus[0] = lnumplustmp;
	astnumplus[0] = astnumplustmp;
	cnumplus[0] = cnumplustmp;
}

__global__ void addKernel(uint8_t * buffer, int** gsyms, int** ggstruct, int** gperm, int** gsign, double* vecl, double pos0, double pos1, double pos2, double vec0, double vec1, double vec2, double addy0, double addy1, double addy2, double addz0, double addz1, double addz2, int rep0, int rep1, int rep2, int rep3, int rep4, int rep5, int* astnum, int* astnumplus, double* astrad, double* astpos0, double* astpos1, double* astpos2, int currenttopo, double* ccpos0, double* ccpos1, double* ccpos2, int* cnumplus, double* lpos0, double* lpos1, double* lpos2, int* lnumplus,bool* rip, int nbframe, double* haz, int* lidx, bool* lout, double* seedist, bool w0, bool w1, bool w2, bool w3, bool w4, bool w5, int* loutf, double m0, double m1, double m2, double m3, double m4, double m5, double m6, double m7, double m8, int* ccsym, int* astsym, double** astmat, int* astplusidx, int* asttopo, int** topo)
	{
	
	double csize = 0.05;
	double lsize = 0.01;
	int i,l;
	double vecn[3]{};
	double nvecn[3]{};
	double coll[3]{};
	double lastcollmin=0;
	double inv[3]{};
	int min;
	double lfac;
	double lcor[3]{};

	double lcol = 0;
	double lcormax;
	int lcorr, lcorg, lcorb;

	double conttmp[3]{};

	int rnbwv;
	int colr, colg, col;

	int tmp = blockIdx.x * blockDim.x + threadIdx.x;
	int tmpx = tmp % 1920;
	int tmpy = (tmp - tmpx) / 1920;
	int ngt[3]{};
	int ngt2[3]{};
	int newsym2;

	int reps[6]{};
	int rand = tmp;
	int noise;
	int wh;
	int uv;
	int coordx, coordy, coord;

	double dark;

	int astcolidx;

	double qa, qb, qc, discr;
	double t1;

	double astcolmin;
	double cpos[3]{};
	int currentsym = 0;
	int newsym;

	double ncpos[3]{};
	double lfe[3]{};
	bool wall[6]{};

	int ccolidx;
	double cont[3]{};
	double ncont0, ncont1, ncont2;
	double u, v;
	double vf;
	int ccor0, ccor1, ccor2;

	wall[0] = w0;
	wall[1] = w1;
	wall[2] = w2;
	wall[3] = w3;
	wall[4] = w4;
	wall[5] = w5;

	for (l = 0; l < 10; l++) rand = (60493 * rand + 11) % 479001599;
	rand += nbframe;
	rand %= 479001599;
	for (l = 0; l < 10; l++) rand = (60493 * rand + 11) % 479001599;

	if (astnum[0] == 0)
	{
		if ((tmpx >= 240 && tmpx <= 240 + 4 * 144 && tmpy >= 135 && tmpy <= 135 + 162) ||
			(tmpx >= 240 && tmpx <= 240 + 144 && tmpy >= 135 && tmpy <= 135 + 5 * 162) ||
			(tmpx >= 240 && tmpx <= 240 + 4 * 144 && tmpy >= 135 + 4 * 162 && tmpy <= 135 + 5 * 162) ||
			(tmpx >= 240 + 3 * 144 && tmpx <= 240 + 4 * 144 && tmpy >= 135 + 2 * 162 && tmpy <= 135 + 5 * 162) ||
			(tmpx >= 240 + 2 * 144 && tmpx <= 240 + 3 * 144 && tmpy >= 135 + 2 * 162 && tmpy <= 135 + 3 * 162) ||

			(tmpx >= 240 + 864 && tmpx <= 240 + 4 * 144 + 864 && tmpy >= 135 && tmpy <= 135 + 162) ||
			(tmpx >= 240 + 864 && tmpx <= 240 + 144 + 864 && tmpy >= 135 && tmpy <= 135 + 5 * 162) ||
			(tmpx >= 240 + 864 && tmpx <= 240 + 4 * 144 + 864 && tmpy >= 135 + 4 * 162 && tmpy <= 135 + 5 * 162) ||
			(tmpx >= 240 + 3 * 144 + 864 && tmpx <= 240 + 4 * 144 + 864 && tmpy >= 135 + 2 * 162 && tmpy <= 135 + 5 * 162) ||
			(tmpx >= 240 + 2 * 144 + 864 && tmpx <= 240 + 3 * 144 + 864 && tmpy >= 135 + 2 * 162 && tmpy <= 135 + 3 * 162))
		{
			rnbwv = rnbw(nbframe);

			colr = rnbwv % 256;
			rnbwv -= colr;
			rnbwv /= 256;
			colg = rnbwv % 256;
			rnbwv -= colg;
			rnbwv /= 256;
			col = rnbwv % 256;

			buffer[4 * tmp] = colr;
			buffer[4 * tmp + 1] = colg;
			buffer[4 * tmp + 2] = col;

			return;
		}

		
	}


	if (rand % 2 == 0) noise = 0;
	else noise = 255;

	if (rip[0])
	{
		if (rand % 2 == 0)
		{
			buffer[4 * tmp] = noise;
			buffer[4 * tmp + 1] = noise;
			buffer[4 * tmp + 2] = noise;
		}
		else
		{
			buffer[4 * tmp] = noise;
			buffer[4 * tmp + 1] = noise;
			buffer[4 * tmp + 2] = noise;
		}

		return;
	}

	for (i = 0; i < 3; i++)
	{
		if (lout[i])
		{
			if (nbframe - loutf[i] > 150) lfe[i] = 4.0 - (nbframe - loutf[i]) / 50.0;
			else lfe[i] = 1.0;
		}
	}


	vecn[0] = vec0 + tmpx * addy0 + tmpy * addz0;
	vecn[1] = vec1 + tmpx * addy1 + tmpy * addz1;
	vecn[2] = vec2 + tmpx * addy2 + tmpy * addz2;

	vecn[0] *= vecl[tmp];
	vecn[1] *= vecl[tmp];
	vecn[2] *= vecl[tmp];

	reps[0] = rep0;
	reps[1] = rep1;
	reps[2] = rep2;
	reps[3] = rep3;
	reps[4] = rep4;
	reps[5] = rep5;

	cpos[0] = pos0 - 0.5;
	cpos[1] = pos1 - 0.5;
	cpos[2] = pos2 - 0.5;
	astcolmin = 2;
	qa = vecn[0] * vecn[0] + vecn[1] * vecn[1] + vecn[2] * vecn[2];
	for (i = 0; i < astnum[0] + astnumplus[0]; i++)
	{
		
		qb = 2 * (vecn[0] * (cpos[0] - astpos0[i]) + vecn[1] * (cpos[1] - astpos1[i]) + vecn[2] * (cpos[2] - astpos2[i]));
		qc = (cpos[0] - astpos0[i]) * (cpos[0] - astpos0[i]) + (cpos[1] - astpos1[i]) * (cpos[1] - astpos1[i]) + (cpos[2] - astpos2[i]) * (cpos[2] - astpos2[i]) - astrad[i] * astrad[i];

		discr = qb * qb - 4 * qa * qc;

		if (discr > 0)
		{
			t1 = ((-1.0) * qb - sqrt(discr)) / (2.0 * qa);
			if (t1 > 0 && t1 < astcolmin)
			{
				astcolmin = t1;
				astcolidx = i;
			}
		}
	}

	for (i = 0; i < cnumplus[0]; i++)
	{
		qb = 2 * (vecn[0] * (cpos[0] - ccpos0[i]) + vecn[1] * (cpos[1] - ccpos1[i]) + vecn[2] * (cpos[2] - ccpos2[i]));
		qc = (cpos[0] - ccpos0[i]) * (cpos[0] - ccpos0[i]) + (cpos[1] - ccpos1[i]) * (cpos[1] - ccpos1[i]) + (cpos[2] - ccpos2[i]) * (cpos[2] - ccpos2[i]) - csize*csize;

		discr = qb * qb - 4 * qa * qc;

		if (discr > 0)
		{
			t1 = ((-1.0) * qb - sqrt(discr)) / (2.0 * qa);
			if (t1 > 0 && t1 < astcolmin)
			{
				astcolmin = t1;
				astcolidx = -1;
				ccolidx = i;
			}
		}
	}
	
	
	ngt[0] = (2 * signbit(vecn[0]) - 1) * -1;
	ngt2[0] = (signbit(vecn[0]) + 1) % 2;
	inv[0] = 1 / vecn[0];
	coll[0] = inv[0] * (ngt2[0] - pos0);
	inv[0] *= ngt[0];

	ngt[1] = (2 * signbit(vecn[1]) - 1) * -1;
	ngt2[1] = (signbit(vecn[1]) + 1) % 2;
	inv[1] = 1 / vecn[1];
	coll[1] = inv[1] * (ngt2[1] - pos1);
	inv[1] *= ngt[1];

	ngt[2] = (2 * signbit(vecn[2]) - 1) * -1;
	ngt2[2] = (signbit(vecn[2]) + 1) % 2;
	inv[2] = 1 / vecn[2];
	coll[2] = inv[2] * (ngt2[2] - pos2);
	inv[2] *= ngt[2];

	if (coll[0] < coll[1]) min = 0;
	else min = 1;
	if (coll[2] < coll[min]) min = 2;

	lcor[0] = 0;
	lcor[1] = 0;
	lcor[2] = 0;

	for (i = 0; i < 3; i++)
	{
		if (lout[i])
		{
			qb = 2 * (vecn[0] * (cpos[0] - lpos0[i]) + vecn[1] * (cpos[1] - lpos1[i]) + vecn[2] * (cpos[2] - lpos2[i]));
			qc = (cpos[0] - lpos0[i]) * (cpos[0] - lpos0[i]) + (cpos[1] - lpos1[i]) * (cpos[1] - lpos1[i]) + (cpos[2] - lpos2[i]) * (cpos[2] - lpos2[i]) - lsize * lsize;

			discr = qb * qb - 4 * qa * qc;

			if (discr > 0)
			{
				t1 = ((-1.0) * qb - sqrt(discr)) / (2.0 * qa);
				if (t1 > 0 && t1 < astcolmin && t1<coll[min])
				{
					lcor[i] += ((-1.0) * qb + sqrt(discr)) / (2.0 * qa) - t1;
				}
			}
		}
	}

	for (i = 0; i < lnumplus[0]; i++)
	{
		qb = 2 * (vecn[0] * (cpos[0] - lpos0[3 + i]) + vecn[1] * (cpos[1] - lpos1[3 + i]) + vecn[2] * (cpos[2] - lpos2[3 + i]));
		qc = (cpos[0] - lpos0[3 + i]) * (cpos[0] - lpos0[3 + i]) + (cpos[1] - lpos1[3 + i]) * (cpos[1] - lpos1[3 + i]) + (cpos[2] - lpos2[3 + i]) * (cpos[2] - lpos2[3 + i]) - lsize * lsize;

		discr = qb * qb - 4 * qa * qc;

		if (discr > 0)
		{
			t1 = ((-1.0) * qb - sqrt(discr)) / (2.0 * qa);
			if (t1 > 0 && t1 < astcolmin && t1 < coll[min])
			{
				lcor[lidx[3 + i]] += ((-1.0) * qb + sqrt(discr)) / (2.0 * qa) - t1;
			}
		}
	}
	
	if (astcolmin < coll[min])
	{
		lcor[0] *= lfe[0];
		lcor[1] *= lfe[1];
		lcor[2] *= lfe[2];

		lcol = lcor[0] + lcor[1] + lcor[2];
		dark = (-1.0 / seedist[0]) * (astcolmin) + 1.0;
		if (dark < 0) dark = 0.0;
		lfac = (1.0 / (4 * lsize * lsize)) * lcol * lcol;
		if (lfac > 1.0) lfac = 1.0;

		if (lcor[0] > lcor[1]) lcormax = lcor[0];
		else lcormax = lcor[1];
		if (lcor[2] > lcormax) lcormax = lcor[2];

		lcorr = 255.0*lcor[0] / lcormax;
		lcorg = 255.0 * lcor[1] / lcormax;
		lcorb = 255.0 * lcor[2] / lcormax;
	
		if (astcolidx == -1)
		{

			newsym = ccsym[ccolidx];

			cont[0] = (cpos[0] + astcolmin * vecn[0]) - ccpos0[ccolidx];
			cont[1] = (cpos[1] + astcolmin * vecn[1]) - ccpos1[ccolidx];
			cont[2] = (cpos[2] + astcolmin * vecn[2]) - ccpos2[ccolidx];

			conttmp[0] = gsign[newsym][0] * cont[gperm[newsym][0]];
			conttmp[1] = gsign[newsym][1] * cont[gperm[newsym][1]];
			conttmp[2] = gsign[newsym][2] * cont[gperm[newsym][2]];

			ncont0 = m0 * conttmp[0]+ m1 * conttmp[1] + m2 * conttmp[2];
			ncont1 = m3 * conttmp[0] + m4 * conttmp[1] + m5 * conttmp[2];
			ncont2 = m6 * conttmp[0] + m7 * conttmp[1] + m8 * conttmp[2];

			ncont0 /= csize;
			ncont1 /= csize;
			ncont2 /= csize;

			u = (0.5 + atan2(ncont1, ncont0) / (2.0 * M_PI));
			v = (0.5 + asin(ncont2) / M_PI);

			rnbwv = rnbw2(u);
			colr = rnbwv % 256;
			rnbwv -= colr;
			rnbwv /= 256;
			colg = rnbwv % 256;
			rnbwv -= colg;
			rnbwv /= 256;
			col = rnbwv % 256;

			if (v > 0.5)
			{
				vf = 2.0 * v - 1.0;

				ccor0 = (1 - vf) * colr + vf * 255.0;
				ccor1 = (1 - vf) * colg + vf * 255.0;
				ccor2 = (1 - vf) * col + vf * 255.0;
			}
			else
			{
				vf = 1.0 - 2.0 * v;

				ccor0 = (1 - vf) * colr;
				ccor1 = (1 - vf) * colg;
				ccor2 = (1 - vf) * col;
			}

			if ((u - 0.5) * (u - 0.5) + (v - 0.5) * (v - 0.5) < 0.0035)
			{
				ccor0 = 128;
				ccor1 = 128;
				ccor2 = 128;
			}
			if ((u - 0.5) * (u - 0.5) + (v - 0.5) * (v - 0.5) < 0.0025)
			{
				ccor0 = 0;
				ccor1 = 0;
				ccor2 = 0;
			}


			buffer[4 * tmp] = haz[0] * noise + (1 - haz[0]) * ((1 - lfac) * dark * (ccor0)+lfac * lcorr);
			buffer[4 * tmp + 1] = haz[0] * noise + (1 - haz[0]) * ((1 - lfac) * dark * (ccor1)+lfac * lcorg);
			buffer[4 * tmp + 2] = haz[0] * noise + (1 - haz[0]) * ((1 - lfac) * dark * (ccor2)+lfac * lcorb);
		}
		else {
			newsym = astsym[astcolidx];
			newsym2 = topo[currenttopo][asttopo[astplusidx[astcolidx]]];

			cont[0] = (cpos[0] + astcolmin * vecn[0]) - astpos0[astcolidx];
			cont[1] = (cpos[1] + astcolmin * vecn[1]) - astpos1[astcolidx];
			cont[2] = (cpos[2] + astcolmin * vecn[2]) - astpos2[astcolidx];

			conttmp[0] = gsign[newsym][0] * cont[gperm[newsym][0]];
			conttmp[1] = gsign[newsym][1] * cont[gperm[newsym][1]];
			conttmp[2] = gsign[newsym][2] * cont[gperm[newsym][2]];

			cont[0] = gsign[newsym2][0] * conttmp[gperm[newsym2][0]];
			cont[1] = gsign[newsym2][1] * conttmp[gperm[newsym2][1]];
			cont[2] = gsign[newsym2][2] * conttmp[gperm[newsym2][2]];


			ncont0 = astmat[astplusidx[astcolidx]][0] * cont[0] + astmat[astplusidx[astcolidx]][1] * cont[1] + astmat[astplusidx[astcolidx]][2] * cont[2];
			ncont1 = astmat[astplusidx[astcolidx]][3] * cont[0] + astmat[astplusidx[astcolidx]][4] * cont[1] + astmat[astplusidx[astcolidx]][5] * cont[2];
			ncont2 = astmat[astplusidx[astcolidx]][6] * cont[0]+ astmat[astplusidx[astcolidx]][7] * cont[1] + astmat[astplusidx[astcolidx]][8] * cont[2];

			ncont0 /= astrad[astplusidx[astcolidx]];
			ncont1 /= astrad[astplusidx[astcolidx]];
			ncont2 /= astrad[astplusidx[astcolidx]];

			u = 100.0*(0.5 + atan2(ncont1, ncont0) / (2.0 * M_PI));
			v = 100.0*(0.5 + asin(ncont2) / M_PI);
			
			uv = (int)u + 100 * (int)v;

			for (l = 0; l < 10; l++) uv = (60493 * uv + 11) % 479001599;

			uv %= 30;
			uv += 128;

			buffer[4 * tmp] = haz[0] * noise + (1 - haz[0]) * ((1 - lfac) * dark * uv + lfac * lcorr);
			buffer[4 * tmp + 1] = haz[0] * noise + (1 - haz[0]) * ((1 - lfac) * dark * uv + lfac * lcorg);
			buffer[4 * tmp + 2] = haz[0] * noise + (1 - haz[0]) * ((1 - lfac) * dark * uv + lfac * lcorb);
		}

		return;
	}

	

	while (coll[min] < seedist[0])
	{
		
		

		cpos[0] = pos0 + vecn[0] * coll[min];
		cpos[1] = pos1 + vecn[1] * coll[min];
		cpos[2] = pos2 + vecn[2] * coll[min];

		if (min == 0)
		{
			if (vecn[0] < 0)
			{
				wh = gsyms[currentsym][3];
				newsym = ggstruct[currentsym][reps[wh]];
				cpos[0] = 0.5;
				cpos[1] = fmod(cpos[1], 1.0);
				if (cpos[1] < 0)cpos[1]++;
				cpos[1] -= 0.5;
				cpos[2] = fmod(cpos[2], 1.0);
				if (cpos[2] < 0)cpos[2]++;
				cpos[2] -= 0.5;
				
			}
			else
			{
				wh = gsyms[currentsym][1];
				newsym = ggstruct[currentsym][reps[wh]];
				cpos[0] = -0.5;
				cpos[1] = fmod(cpos[1], 1.0);
				if (cpos[1] < 0)cpos[1]++;
				cpos[1] -= 0.5;
				cpos[2] = fmod(cpos[2], 1.0);
				if (cpos[2] < 0)cpos[2]++;
				cpos[2] -= 0.5;
				
			}
		}
		else if (min == 1)
		{
			if (vecn[1] < 0)
			{
				wh = gsyms[currentsym][4];
				newsym = ggstruct[currentsym][reps[wh]];
				cpos[0] = fmod(cpos[0], 1.0);
				if (cpos[0] < 0)cpos[0]++;
				cpos[0] -= 0.5;
				cpos[1] = 0.5;
				cpos[2] = fmod(cpos[2], 1.0);
				if (cpos[2] < 0)cpos[2]++;
				cpos[2] -= 0.5;
				

			}
			else
			{
				wh = gsyms[currentsym][2];
				newsym = ggstruct[currentsym][reps[wh]];
				cpos[0] = fmod(cpos[0], 1.0);
				if (cpos[0] < 0)cpos[0]++;
				cpos[0] -= 0.5;
				cpos[1] = -0.5;
				cpos[2] = fmod(cpos[2], 1.0);
				if (cpos[2] < 0)cpos[2]++;
				cpos[2] -= 0.5;
				
			}
		}
		else 
		{
			if (vecn[2] < 0)
			{
				wh = gsyms[currentsym][5];
				newsym = ggstruct[currentsym][reps[wh]];
				cpos[0] = fmod(cpos[0], 1.0);
				if (cpos[0] < 0)cpos[0]++;
				cpos[0] -= 0.5;
				cpos[1] = fmod(cpos[1], 1.0);
				if (cpos[1] < 0)cpos[1]++;
				cpos[1] -= 0.5;
				cpos[2] = 0.5;
				
			}
			else
			{
				wh = gsyms[currentsym][0];
				newsym = ggstruct[currentsym][reps[wh]];
				cpos[0] = fmod(cpos[0], 1.0);
				if (cpos[0] < 0)cpos[0]++;
				cpos[0] -= 0.5;
				cpos[1] = fmod(cpos[1], 1.0);
				if (cpos[1] < 0)cpos[1]++;
				cpos[1] -= 0.5;
				cpos[2] = -0.5;
				
			}
		}

		ncpos[0] = gsign[newsym][0] * cpos[gperm[newsym][0]];
		ncpos[1] = gsign[newsym][1] * cpos[gperm[newsym][1]];
		ncpos[2] = gsign[newsym][2] * cpos[gperm[newsym][2]];

		nvecn[0] = gsign[newsym][0] * vecn[gperm[newsym][0]];
		nvecn[1] = gsign[newsym][1] * vecn[gperm[newsym][1]];
		nvecn[2] = gsign[newsym][2] * vecn[gperm[newsym][2]];

		

		if (wall[wh])
		{
			lcor[0] *= lfe[0];
			lcor[1] *= lfe[1];
			lcor[2] *= lfe[2];

			lcol = lcor[0] + lcor[1] + lcor[2];
			dark = (-1.0 / seedist[0]) * (coll[min]) + 1.0;
			if (dark < 0) dark = 0.0;
			lfac = (1.0 / (4 * lsize * lsize)) * lcol * lcol;
			if (lfac > 1.0) lfac = 1.0;

			if (lcor[0] > lcor[1]) lcormax = lcor[0];
			else lcormax = lcor[1];
			if (lcor[2] > lcormax) lcormax = lcor[2];

			lcorr = 255.0 * lcor[0] / lcormax;
			lcorg = 255.0 * lcor[1] / lcormax;
			lcorb = 255.0 * lcor[2] / lcormax;

			if (ncpos[0] == 0.5 || ncpos[0] == -0.5)
			{
				coordx = 100*(ncpos[1]+0.5);
				coordy = 100 * (ncpos[2]+0.5);
			}
			else if (ncpos[1] == 0.5 || ncpos[1] == -0.5)
			{
				coordx = 100 * (ncpos[0]+0.5);
				coordy = 100 * (ncpos[2]+0.5);
			}
			else
			{
				coordx = 100 * (ncpos[0]+0.5);
				coordy = 100 * (ncpos[1]+0.5);
			}

			coord = coordx + 100 * coordy;

			for (l = 0; l < 10; l++) coord = (60493 * coord + 11) % 479001599;
			coord %= 20;
			coord += 45;

			
				buffer[4 * tmp] = haz[0] * noise + (1 - haz[0]) * ((1 - lfac) * dark * 2.0*coord+ lfac * lcorr);
				buffer[4 * tmp + 1] = haz[0] * noise + (1 - haz[0]) * ((1 - lfac) * dark *coord + lfac * lcorg);
				buffer[4 * tmp + 2] = haz[0] * noise + (1 - haz[0]) * ( lfac * lcorb);
			
			

			return;
		}
	
		
	


		astcolmin = 2;
		qa = nvecn[0] * nvecn[0] + nvecn[1] * nvecn[1] + nvecn[2] * nvecn[2];
		for (i = 0; i < astnum[0] + astnumplus[0]; i++)
		{
			
			qb = 2 * (nvecn[0] * (ncpos[0] - astpos0[i]) + nvecn[1] * (ncpos[1] - astpos1[i]) + nvecn[2] * (ncpos[2] - astpos2[i]));
			qc = (ncpos[0] - astpos0[i]) * (ncpos[0] - astpos0[i]) + (ncpos[1] - astpos1[i]) * (ncpos[1] - astpos1[i]) + (ncpos[2] - astpos2[i]) * (ncpos[2] - astpos2[i]) - astrad[i] * astrad[i];
			
			discr = qb * qb - 4 * qa * qc;

	

			if (discr > 0)
			{
				t1 = ((-1.0) * qb - sqrt(discr)) / (2.0 * qa);
				if (t1 > 0 && t1<astcolmin)
				{
						astcolmin = t1;
						astcolidx = i;
				}
			}
		}

		for (i = 0; i < cnumplus[0]; i++)
		{
			qb = 2 * (nvecn[0] * (ncpos[0] - ccpos0[i]) + nvecn[1] * (ncpos[1] - ccpos1[i]) + nvecn[2] * (ncpos[2] - ccpos2[i]));
			qc = (ncpos[0] - ccpos0[i]) * (ncpos[0] - ccpos0[i]) + (ncpos[1] - ccpos1[i]) * (ncpos[1] - ccpos1[i]) + (ncpos[2] - ccpos2[i]) * (ncpos[2] - ccpos2[i]) - csize * csize;

			discr = qb * qb - 4 * qa * qc;


			if (discr > 0)
			{
				t1 = ((-1.0) * qb - sqrt(discr)) / (2.0 * qa);
				if (t1 > 0 && t1 < astcolmin)
				{
					astcolmin = t1;
					astcolidx = -1;
					ccolidx = i;
				}
			}
		}

		
		
		currentsym = newsym;

		lastcollmin = coll[min];
		coll[min] += inv[min];

		if (coll[0] < coll[1]) min = 0;
		else min = 1;
		if (coll[2] < coll[min]) min = 2;

		for (i = 0; i < 3; i++)
		{
			if (lout[i])
			{
				qb = 2 * (nvecn[0] * (ncpos[0] - lpos0[i]) + nvecn[1] * (ncpos[1] - lpos1[i]) + nvecn[2] * (ncpos[2] - lpos2[i]));
				qc = (ncpos[0] - lpos0[i]) * (ncpos[0] - lpos0[i]) + (ncpos[1] - lpos1[i]) * (ncpos[1] - lpos1[i]) + (ncpos[2] - lpos2[i]) * (ncpos[2] - lpos2[i]) - lsize * lsize;

				discr = qb * qb - 4 * qa * qc;


				if (discr > 0)
				{
					t1 = ((-1.0) * qb - sqrt(discr)) / (2.0 * qa);
					if (t1 > 0 && t1 < astcolmin && t1<coll[min]-lastcollmin)
					{
						lcor[i] += ((-1.0) * qb + sqrt(discr)) / (2.0 * qa) - t1;
					}
				}
			}
		}

		for (i = 0; i < lnumplus[0]; i++)
		{
			qb = 2 * (nvecn[0] * (ncpos[0] - lpos0[3 + i]) + nvecn[1] * (ncpos[1] - lpos1[3 + i]) + nvecn[2] * (ncpos[2] - lpos2[3 + i]));
			qc = (ncpos[0] - lpos0[3 + i]) * (ncpos[0] - lpos0[3 + i]) + (ncpos[1] - lpos1[3 + i]) * (ncpos[1] - lpos1[3 + i]) + (ncpos[2] - lpos2[3 + i]) * (ncpos[2] - lpos2[3 + i]) - lsize * lsize;

			discr = qb * qb - 4 * qa * qc;


			if (discr > 0)
			{
				t1 = ((-1.0) * qb - sqrt(discr)) / (2.0 * qa);
				if (t1 > 0 && t1 < astcolmin && t1 < coll[min] - lastcollmin)
				{
					lcor[lidx[3 + i]] += ((-1.0) * qb + sqrt(discr)) / (2.0 * qa) - t1;
				}
			}
		}

		if (astcolmin < coll[min] - lastcollmin)
		{
			lcor[0] *= lfe[0];
			lcor[1] *= lfe[1];
			lcor[2] *= lfe[2];

			lcol = lcor[0] + lcor[1] + lcor[2];
			dark = (-1.0 / seedist[0]) * (astcolmin + lastcollmin) + 1.0;
			if (dark < 0) dark = 0.0;
			lfac = (1.0 / (4 * lsize * lsize)) * lcol * lcol;
			if (lfac > 1.0) lfac = 1.0;

			if (lcor[0] > lcor[1]) lcormax = lcor[0];
			else lcormax = lcor[1];
			if (lcor[2] > lcormax) lcormax = lcor[2];

			lcorr = 255.0 * lcor[0] / lcormax;
			lcorg = 255.0 * lcor[1] / lcormax;
			lcorb = 255.0 * lcor[2] / lcormax;

			

			if (astcolidx == -1)
			{
					newsym = ccsym[ccolidx];

					cont[0] = (ncpos[0] + astcolmin * nvecn[0]) - ccpos0[ccolidx];
					cont[1] = (ncpos[1] + astcolmin * nvecn[1]) - ccpos1[ccolidx];
					cont[2] = (ncpos[2] + astcolmin * nvecn[2]) - ccpos2[ccolidx];

					conttmp[0] = gsign[newsym][0] * cont[gperm[newsym][0]];
					conttmp[1] = gsign[newsym][1] * cont[gperm[newsym][1]];
					conttmp[2] = gsign[newsym][2] * cont[gperm[newsym][2]];

					ncont0 = m0 * conttmp[0] + m1 * conttmp[1] + m2 * conttmp[2];
					ncont1 = m3 * conttmp[0] + m4 * conttmp[1] + m5 * conttmp[2];
					ncont2 = m6 * conttmp[0] + m7 * conttmp[1] + m8 * conttmp[2];

					ncont0 /= csize;
					ncont1 /= csize;
					ncont2 /= csize;

					u =(0.5 + atan2(ncont1, ncont0) / (2.0 * M_PI));
					v = (0.5 + asin(ncont2) / M_PI);
		
					rnbwv = rnbw2(u);
					colr = rnbwv % 256;
					rnbwv -= colr;
					rnbwv /= 256;
					colg = rnbwv % 256;
					rnbwv -= colg;
					rnbwv /= 256;
					col = rnbwv % 256;
					
					if (v > 0.5)
					{
						vf = 2.0 * v - 1.0;

						ccor0 = (1-vf)*colr + vf*255.0;
						ccor1 = (1 - vf) * colg + vf * 255.0;
						ccor2 = (1 - vf) * col + vf * 255.0;
					}
					else
					{
						vf = 1.0 - 2.0 * v;

						ccor0 = (1 - vf) * colr;
						ccor1 = (1 - vf) * colg;
						ccor2 = (1 - vf) * col;
					}

					if ((u-0.5)*(u-0.5) + (v-0.5)*(v-0.5)<0.0035)
					{
						ccor0 = 128;
						ccor1 = 128;
						ccor2 = 128;
					}
					if ((u - 0.5) * (u - 0.5) + (v - 0.5) * (v - 0.5) < 0.0025)
					{
						ccor0 = 0;
						ccor1 = 0;
						ccor2 = 0;
					}


					buffer[4 * tmp] = haz[0] * noise + (1 - haz[0]) * ((1 - lfac) * dark * (ccor0) + lfac * lcorr);
					buffer[4 * tmp + 1] = haz[0] * noise + (1 - haz[0]) * ((1 - lfac) * dark * (ccor1)+lfac * lcorg);
					buffer[4 * tmp + 2] = haz[0] * noise + (1 - haz[0]) * ((1 - lfac) * dark * (ccor2)+lfac * lcorb);

				
			}
			else {
				newsym = astsym[astcolidx];
				newsym2 = topo[currenttopo][asttopo[astplusidx[astcolidx]]];

				cont[0] = (ncpos[0] + astcolmin * nvecn[0]) - astpos0[astcolidx];
				cont[1] = (ncpos[1] + astcolmin * nvecn[1]) - astpos1[astcolidx];
				cont[2] = (ncpos[2] + astcolmin * nvecn[2]) - astpos2[astcolidx];

				conttmp[0] = gsign[newsym][0] * cont[gperm[newsym][0]];
				conttmp[1] = gsign[newsym][1] * cont[gperm[newsym][1]];
				conttmp[2] = gsign[newsym][2] * cont[gperm[newsym][2]];

				cont[0] = gsign[newsym2][0] * conttmp[gperm[newsym2][0]];
				cont[1] = gsign[newsym2][1] * conttmp[gperm[newsym2][1]];
				cont[2] = gsign[newsym2][2] * conttmp[gperm[newsym2][2]];


				ncont0 = astmat[astplusidx[astcolidx]][0] * cont[0] + astmat[astplusidx[astcolidx]][1] * cont[1] + astmat[astplusidx[astcolidx]][2] * cont[2];
				ncont1 = astmat[astplusidx[astcolidx]][3] * cont[0] + astmat[astplusidx[astcolidx]][4] * cont[1] + astmat[astplusidx[astcolidx]][5] * cont[2];
				ncont2 = astmat[astplusidx[astcolidx]][6] * cont[0] + astmat[astplusidx[astcolidx]][7] * cont[1] + astmat[astplusidx[astcolidx]][8] * cont[2];

				ncont0 /= astrad[astplusidx[astcolidx]];
				ncont1 /= astrad[astplusidx[astcolidx]];
				ncont2 /= astrad[astplusidx[astcolidx]];

				u = 100.0 * (0.5 + atan2(ncont1, ncont0) / (2.0 * M_PI));
				v = 100.0 * (0.5 + asin(ncont2) / M_PI);

				uv = (int)u + 100 * (int)v;

				for (l = 0; l < 10; l++) uv = (60493 * uv + 11) % 479001599;

				uv %= 30;
				uv += 128;

				buffer[4 * tmp] = haz[0] * noise + (1 - haz[0]) * ((1 - lfac) * dark * uv + lfac * lcorr);
				buffer[4 * tmp + 1] = haz[0] * noise + (1 - haz[0]) * ((1 - lfac) * dark * uv + lfac * lcorg);
				buffer[4 * tmp + 2] = haz[0] * noise + (1 - haz[0]) * ((1 - lfac) * dark * uv + lfac * lcorb);
			}

			return;
		}
	}

	lcor[0] *= lfe[0];
	lcor[1] *= lfe[1];
	lcor[2] *= lfe[2];

	lcol = lcor[0] + lcor[1] + lcor[2];
	lfac = (1.0 / (4 * lsize * lsize)) * lcol * lcol;
	if (lfac > 1.0) lfac = 1.0;

	if (lcor[0] > lcor[1]) lcormax = lcor[0];
	else lcormax = lcor[1];
	if (lcor[2] > lcormax) lcormax = lcor[2];

	lcorr = 255.0 * lcor[0] / lcormax;
	lcorg = 255.0 * lcor[1] / lcormax;
	lcorb = 255.0 * lcor[2] / lcormax;

	buffer[4 * tmp] = haz[0] * noise + (1 - haz[0]) * (lcorr * lfac);
	buffer[4 * tmp + 1] = haz[0] * noise + (1 - haz[0]) * (lcorg*lfac);
	buffer[4 * tmp + 2] = haz[0] * noise + (1 - haz[0]) * (lcorb * lfac);


}


void cudaInit(int syms[48][6], int gstruct[48][48], int perm[48][3], int sign[48][3], int gtopo[187][1728], int ginv[48])
{
	
	int i;
	double disto = 2;
	double sqsz = 0.01 / 6;
	int tmpx, tmpy;

	double* vecltmp = new double[1920 * 1080];

	double vec0, vec1, vec2;
	double addy0, addy1, addy2;
	double addz0, addz1, addz2;
	double vecn0, vecn1, vecn2;
	double x00 = 1, x01 = 0, x02 = 0;
	double x10 = 0, x11 = 1, x12 = 0;
	double x20 = 0, x21 = 0, x22 = 1;
	double multy = (1 - 1920) * sqsz / 2;
	double multz = (1080 - 1) * sqsz / 2;

	int* hsyms[187]{};
	double* hhsyms[64]{};

	cudaSetDevice(0);
	cudaMalloc((void**)&buffer, 4 * 1920 * 1080 * sizeof(uint8_t));
	cudaMalloc((void**)&vecl, 1920 * 1080 * sizeof(double));
	
	cudaMalloc((void**)&gsyms,  48* sizeof(int*));
	for (i = 0; i < 48; i++) cudaMalloc((void**)&hsyms[i], 6 * sizeof(int));
	cudaMemcpy(gsyms, hsyms, 48 * sizeof(int*), cudaMemcpyHostToDevice);
	for (i = 0; i < 48; i++)  cudaMemcpy(hsyms[i], syms[i], 6 * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&ggstruct, 48 * sizeof(int*));
	for (i = 0; i < 48; i++) cudaMalloc((void**)&hsyms[i], 48 * sizeof(int));
	cudaMemcpy(ggstruct, hsyms, 48 * sizeof(int*), cudaMemcpyHostToDevice);
	for (i = 0; i < 48; i++)  cudaMemcpy(hsyms[i], gstruct[i], 48 * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&gperm, 48 * sizeof(int*));
	for (i = 0; i < 48; i++) cudaMalloc((void**)&hsyms[i], 3 * sizeof(int));
	cudaMemcpy(gperm, hsyms, 48 * sizeof(int*), cudaMemcpyHostToDevice);
	for (i = 0; i < 48; i++)  cudaMemcpy(hsyms[i], perm[i], 3 * sizeof(int), cudaMemcpyHostToDevice);
	
	cudaMalloc((void**)&gsign, 48 * sizeof(int*));
	for (i = 0; i < 48; i++) cudaMalloc((void**)&hsyms[i], 3 * sizeof(int));
	cudaMemcpy(gsign, hsyms, 48 * sizeof(int*), cudaMemcpyHostToDevice);
	for (i = 0; i < 48; i++)  cudaMemcpy(hsyms[i], sign[i], 3 * sizeof(int), cudaMemcpyHostToDevice);
	
	cudaMalloc((void**)&topo, 187 * sizeof(int*));
	for (i = 0; i < 187; i++) cudaMalloc((void**)&hsyms[i], 1728 * sizeof(int));
	cudaMemcpy(topo, hsyms, 187 * sizeof(int*), cudaMemcpyHostToDevice);
	for (i = 0; i < 187; i++)  cudaMemcpy(hsyms[i], gtopo[i], 1728 * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&astmat, 64 * sizeof(double*));
	for (i = 0; i < 64; i++) cudaMalloc((void**)&hhsyms[i], 9 * sizeof(double));
	cudaMemcpy(astmat, hhsyms, 64 * sizeof(double*), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&astmatv, 64 * sizeof(double*));
	for (i = 0; i < 64; i++) cudaMalloc((void**)&hhsyms[i], 9 * sizeof(double));
	cudaMemcpy(astmatv, hhsyms, 64 * sizeof(double*), cudaMemcpyHostToDevice);
	

	vec0 = disto * x00 + multy * x10 + multz * x20;
	vec1 = disto * x01 + multy * x11 + multz * x21;
	vec2 = disto * x02 + multy * x12 + multz * x22;

	addy0 = sqsz * x10;
	addy1 = sqsz * x11;
	addy2 = sqsz * x12;

	addz0 = -sqsz * x20;
	addz1 = -sqsz * x21;
	addz2 = -sqsz * x22;

	for (i = 0; i < 1920 * 1080; i++)
	{
		tmpx = i % 1920;
		tmpy = (i - tmpx) / 1920;

		vecn0 = vec0 + tmpx * addy0 + tmpy * addz0;
		vecn1 = vec1 + tmpx * addy1 + tmpy * addz1;
		vecn2 = vec2 + tmpx * addy2 + tmpy * addz2;

		vecltmp[i] = 1.0/sqrt(vecn0 * vecn0 + vecn1 * vecn1 + vecn2 * vecn2);
	}

	cudaMemcpy(vecl, vecltmp, 1920 * 1080 * sizeof(double), cudaMemcpyHostToDevice);

	

	cudaMalloc((void**)&astrad, 64 * sizeof(double));
	cudaMalloc((void**)&asttopo, 64 * sizeof(int));
	cudaMalloc((void**)&astpos0, 64 * sizeof(double));
	cudaMalloc((void**)&astpos1, 64 * sizeof(double));
	cudaMalloc((void**)&astpos2, 64 * sizeof(double));
	cudaMalloc((void**)&astv0, 64 * sizeof(double));
	cudaMalloc((void**)&astv1, 64 * sizeof(double));
	cudaMalloc((void**)&astv2, 64 * sizeof(double));
	cudaMalloc((void**)&astnumplus, sizeof(int));
	cudaMalloc((void**)&cnumplus, sizeof(int));
	cudaMalloc((void**)&ccpos0, 10 * sizeof(double));
	cudaMalloc((void**)&ccpos1, 10 * sizeof(double));
	cudaMalloc((void**)&ccpos2, 10 * sizeof(double));
	cudaMalloc((void**)&ccsym, 10 * sizeof(int));
	cudaMalloc((void**)&astsym, 64 * sizeof(int));

	cudaMalloc((void**)&lnumplus, sizeof(int));
	cudaMalloc((void**)&lpos0, 64 * sizeof(double));
	cudaMalloc((void**)&lpos1, 64 * sizeof(double));
	cudaMalloc((void**)&lpos2, 64 * sizeof(double));
	cudaMalloc((void**)&lv0, 64 * sizeof(double));
	cudaMalloc((void**)&lv1, 64 * sizeof(double));
	cudaMalloc((void**)&lv2, 64 * sizeof(double));
	cudaMalloc((void**)&lout, 64 * sizeof(bool));
	cudaMalloc((void**)&loutf, 64 * sizeof(int));
	cudaMalloc((void**)&lidx, 64 * sizeof(int));
	cudaMalloc((void**)&astnum, sizeof(int));
	cudaMalloc((void**)&astplusidx, 64 * sizeof(int));
	cudaMalloc((void**)&rip, sizeof(bool));
	cudaMalloc((void**)&haz, sizeof(double));
	cudaMalloc((void**)&seedist, sizeof(double));
	cudaMalloc((void**)&dist, sizeof(double));
	cudaMalloc((void**)&gginv, 48*sizeof(int));
	cudaMemcpy(gginv, ginv, 48 * sizeof(int), cudaMemcpyHostToDevice);

	setast << <1, 1 >> > (buffer, astrad, astpos0, astpos1, astpos2,astv0,astv1,astv2,astnum,seedist,dist,astmat,astmatv);
	cudaDeviceSynchronize();
}

void cudaExit()
{
	cudaFree(buffer);
	cudaFree(vecl);
	cudaFree(astrad);
	cudaFree(astpos0);
	cudaFree(astpos1);
	cudaFree(astpos2);
	cudaFree(astv0);
	cudaFree(astv1);
	cudaFree(astv2);
	cudaFree(astnumplus);
	cudaFree(cnumplus);
	cudaFree(ccpos0);
	cudaFree(ccpos1);
	cudaFree(ccpos2);
	cudaFree(lnumplus);
	cudaFree(lpos0);
	cudaFree(lpos1);
	cudaFree(lpos2);
	cudaFree(lv0);
	cudaFree(lv1);
	cudaFree(lv2);
	cudaFree(lout);
	cudaFree(loutf);
	cudaFree(astnum);
	cudaFree(astplusidx);
	cudaFree(gginv);
	cudaFree(seedist);
	cudaFree(dist);
	cudaFree(ccsym);
	cudaFree(astsym);
	cudaFree(astmat);
	cudaFree(astmatv);
	cudaDeviceReset();
}

void cudathingy(uint8_t* pixels, double pos0, double pos1, double pos2, double vec0, double vec1, double vec2, double addy0, double addy1, double addy2, double addz0, double addz1, double addz2, int rep0, int rep1, int rep2, int rep3, int rep4, int rep5, int currenttopo, bool fireflag, double v0, double v1, double v2, int nbframe, bool resetflag, bool sdp, bool sdm, bool fovp, bool fovm, double cv0, double cv1, double cv2, bool w0, bool w1, bool w2, bool w3, bool w4, bool w5, double m0, double m1, double m2, double m3, double m4, double m5, double m6, double m7, double m8)
{
	if (fovp || fovm || resetflag)
	{
		modfovp << <1, 1>>> (dist, fovp, fovm, resetflag);
		cudaDeviceSynchronize();
		modfov << <(int)(1920 * 1080 / 480), 480 >> > (dist, vecl);
		cudaDeviceSynchronize();
	}
	rotast << <1, 32 >> > (astmat, astmatv);
	cudaDeviceSynchronize();
	modast << <1, 1 >> > (astrad, astpos0, astpos1, astpos2,astv0,astv1,astv2,topo,gperm,gsign,currenttopo,astnum,astnumplus, lpos0,lpos1,lpos2,lv0,lv1,lv2,lnumplus, pos0, pos1, pos2, ccpos0, ccpos1, ccpos2, cnumplus, fireflag, v0, v1, v2, lout, loutf, nbframe,gginv,astplusidx,rip,haz,lidx,resetflag,seedist,sdp,sdm,cv0,cv1,cv2,w0,w1,w2,w3,w4,w5,ccsym,astsym,astmat,astmatv,asttopo);
	cudaDeviceSynchronize();
	addKernel << <(int)(1920 * 1080 / 480), 480 >> > (buffer, gsyms, ggstruct, gperm, gsign, vecl, pos0, pos1, pos2, vec0, vec1, vec2, addy0, addy1, addy2, addz0, addz1, addz2,rep0,rep1,rep2,rep3,rep4,rep5, astnum, astnumplus, astrad,  astpos0,  astpos1,  astpos2,currenttopo,ccpos0, ccpos1,  ccpos2, cnumplus, lpos0, lpos1, lpos2, lnumplus,rip,nbframe,haz,lidx,lout,seedist,w0,w1,w2,w3,w4,w5,loutf,m0,m1,m2,m3,m4,m5,m6,m7,m8,ccsym,astsym,astmat,astplusidx,asttopo,topo);
	cudaDeviceSynchronize();
	cudaMemcpy(pixels, buffer, 4 * 1920 * 1080 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
}