#define _USE_MATH_DEFINES

#include <SFML/Graphics.hpp>
#include "windows.h" 
#include <stdio.h>


static int lat[50][50][50]{};
static bool latset[50][50][50]{};
static int syms[48][6]{};
static int symsinv[48][6]{};
static int gstruct[48][48]{};
static int ginv[48]{};
static int reps[187][6]{};
static int topo[187][12][12][12]{};
static int gtopo[187][1728]{};

static int perm[48][3];
static int sign[48][3];

static int permb[48][3];
static int signb[48][3];

void cudathingy(uint8_t* pixels, double pos0, double pos1, double pos2, double vec0, double vec1, double vec2, double addy0, double addy1, double addy2, double addz0, double addz1, double addz2, int rep0, int rep1, int rep2, int rep3, int rep4, int rep5, int currenttopo, bool fireflag, double v0,double v1, double v2, int nbframe, bool resetflag, bool sdp, bool sdm,bool fovp, bool fovm, double cv0, double cv1, double cv2, bool w0, bool w1, bool w2, bool w3, bool w4, bool w5, double m0,double m1,double m2, double m3, double m4, double m5, double m6, double m7, double m8);
void cudaInit(int syms[48][6], int gstruct[48][48],int perm[48][3], int sign[48][3], int gtopo[187][1728], int ginv[48]);
void cudaExit();

int main()
{
	ShowWindow(GetConsoleWindow(), SW_HIDE);
	//ShowWindow(GetConsoleWindow(), SW_SHOW);

	int i, j,k,l,tmp,tmp2,rem;
	int currenttopo = 0;
	int a, b, c, d, e, f;
	int neigh[6]{};
	bool list[6]{};
	bool neighset[6]{};
	int ppos0, ppos1, ppos2;
	int newsym;
	double postmp[3]{};
	int nbframe = 0;
	int legit = 0;
	int max;
	int rep0, rep1, rep2, rep3, rep4, rep5;
	double cv[3]{};
	double newcv[3]{};
	int opp[6]{};
	int tmpvec[6]{};
	int coords[6]{};
	double det, m0, m1, m2, m3, m4, m5, m6, m7, m8;

	double tmpd;
	double csize = 0.05;
	bool sdp=false, sdm=false;

	bool walls[6]{};

	
	opp[0] = 5;
	opp[1] = 3;
	opp[2] = 4;
	opp[3] = 1;
	opp[4] = 2;
	opp[5] = 0;

	coords[0] = 2;
	coords[1] = 0;
	coords[2] = 1;
	coords[3] = 0;
	coords[4] = 1;
	coords[5] = 2;

	double dist = 2;
	double sqsz = 0.01 / 4;
	double speed = 0.0001;

	int mousx, mousy, centralx, centraly;

	double pos0 = 0.5, pos1 = 0.5, pos2 = 0.03;
	double npos0, npos1, npos2;
	double vec0, vec1, vec2;
	double addy0, addy1, addy2;
	double addz0, addz1, addz2;
	double multy = (1 - 1920) * sqsz / 2;
	double multz = (1080 - 1) * sqsz / 2;

	bool fireflag=false;
	bool resetflag = false;
	double x[3][3]{};
	double newx[3][3]{};
	x[0][0] = 1;
	x[1][1] = 1;
	x[2][2] = 1;

	double anglex, angley;
	bool focus = true;
	bool fovm = false;
	bool fovp = false;

	int wp[3][2]{};
	int walltmpidx;
	bool walltmp[6]{};

	wp[0][0] = 0;
	wp[0][1] = 5;
	wp[1][0] = 1;
	wp[1][1] = 3;
	wp[2][0] = 2;
	wp[2][1] = 4;

	
	for (i = 0; i < 6; i++)
	{
		for (j = 0; j < 4; j++)
		{
			for (k = 0; k < 2; k++)
			{
				for (l = 0; l < 6; l++) list[l] = true;
				tmp = i + 6 * j + 24 * k;
				syms[tmp][0] = i;
				syms[tmp][5] = opp[i];
				symsinv[tmp][i] = 0;
				symsinv[tmp][opp[i]] =5;
				list[i] = false;
				list[opp[i]] = false;

				tmp2 = j;
				for (l = 0; l < 6; l++)
				{
					if (list[l])
					{
						tmp2--;
						if (tmp2 < 0)
						{
							rem = l;
							l = 6;
						}
					}
				}
				
				syms[tmp][1] = rem;
				syms[tmp][3] = opp[rem];
				symsinv[tmp][rem] = 1;
				symsinv[tmp][opp[rem]] = 3;
				list[rem] = false;
				list[opp[rem]] = false;
	
				tmp2 = k;
				for (l = 0; l < 6; l++)
				{
					if (list[l])
					{
						tmp2--;
						if (tmp2 < 0)
						{
							rem = l;
							l = 6;
						}
					}
				}

				syms[tmp][2] = rem;
				syms[tmp][4] = opp[rem];
				symsinv[tmp][rem] = 2;
				symsinv[tmp][opp[rem]] =4;
			}
		}
	}
	for (i = 0; i < 48; i++)
	{
		for (j = 0; j < 48; j++)
		{
			for (k = 0; k < 6; k++)
			{
				tmpvec[k] = syms[j][syms[i][k]];
			}

			for (k = 0; k < 48; k++)
			{
				for (l = 0; l < 6; l++)
				{
					if (tmpvec[l] != syms[k][l]) l = 7;
				}
				if (l == 6)
				{
					tmp = k;
					k = 49;
				}
			}

			gstruct[i][j] = tmp;
		}
	}
	for (i = 0; i < 48; i++)
	{
		for (j = 0; j < 48; j++)
		{
			if (gstruct[i][j] == 0)
			{
				ginv[i] = j;
				j = 49;
			}
		}
	}

	for (a = 0; a < 48; a++)
	{
		for (b = 0; b < 48; b++)
		{
			for (c = 0; c < 48; c++)
			{
				for (i = 1; i < 6; i++) neighset[i] = false;
				neigh[0] = a;
				
				tmp = syms[a][5];
				neigh[tmp] = ginv[a];
				neighset[tmp] = true;

				i = 1;
				while (neighset[i]) i++;

				neigh[i] = b;
				neighset[i] = true;

				tmp = syms[b][opp[i]];
				neigh[tmp] = ginv[b];
				neighset[tmp] = true;

				i = 2;
				while (neighset[i]) i++;

				neigh[i] = c;
				neighset[i] = true;

				tmp = syms[c][opp[i]];
				neigh[tmp] = ginv[c];
				neighset[tmp] = true;

				


				if (neighset[1] && neighset[2] && neighset[3] && neighset[4] && neighset[5])
				{

					for (i = 0; i < legit; i++)
					{
						for (j = 1; j < 48; j++)
						{
							for (k = 0; k < 6; k++) if (neigh[syms[j][k]] != reps[i][k]) k = 6;
							if (k == 6) i = legit;
						}
					}

					if (i == legit)
					{

						for (i = 0; i < 12; i++)
						{
							for (j = 0; j < 50; j++)
							{
								for (k = 0; k < 50; k++)
								{
									lat[i][j][k] = 0;
									latset[i][j][k] = false;
								}
							}
						}
						latset[0][0][0] = true;

						for (i = 0; i < 12; i++)
						{
							for (j = 0; j < 12; j++)
							{
								for (k = 0; k < 12; k++)
								{
									if (i < 11)
									{
										tmp = gstruct[lat[i][j][k]][neigh[syms[lat[i][j][k]][0]]];
										if (latset[i + 1][j][k] && lat[i + 1][j][k] != tmp)
										{
											k = 13; j = 13; i = 13;
										}
										else
										{
											latset[i + 1][j][k] = true;
											lat[i + 1][j][k] = tmp;
										}
									}

									if (k < 11 && k != 13)
									{
										tmp = gstruct[lat[i][j][k]][neigh[syms[lat[i][j][k]][1]]];
										if (latset[i][j][k + 1] && lat[i][j][k + 1] != tmp)
										{
											k = 13; j = 13; i = 13;
										}
										else
										{
											latset[i][j][k + 1] = true;
											lat[i][j][k + 1] = tmp;
										}
									}
									if (j < 11 && k != 13)
									{
										tmp = gstruct[lat[i][j][k]][neigh[syms[lat[i][j][k]][2]]];
										if (latset[i][j + 1][k] && lat[i][j + 1][k] != tmp)
										{
											k = 13; j = 13; i = 13;
										}
										else
										{
											latset[i][j + 1][k] = true;
											lat[i][j + 1][k] = tmp;
										}
									}
									if (k > 0 && k != 13)
									{
										tmp = gstruct[lat[i][j][k]][neigh[syms[lat[i][j][k]][3]]];
										if (latset[i][j][k - 1] && lat[i][j][k - 1] != tmp)
										{
											k = 13; j = 13; i = 13;
										}
										else
										{
											latset[i][j][k - 1] = true;
											lat[i][j][k - 1] = tmp;
										}
									}
									if (j > 0 && k != 13)
									{
										tmp = gstruct[lat[i][j][k]][neigh[syms[lat[i][j][k]][4]]];
										if (latset[i][j - 1][k] && lat[i][j - 1][k] != tmp)
										{
											k = 13; j = 13; i = 13;
										}
										else
										{
											latset[i][j - 1][k] = true;
											lat[i][j - 1][k] = tmp;
										}
									}
									if (i > 0 && k != 13)
									{
										tmp = gstruct[lat[i][j][k]][neigh[syms[lat[i][j][k]][5]]];
										if (latset[i - 1][j][k] && lat[i - 1][j][k] != tmp)
										{
											k = 13; j = 13; i = 13;
										}
										else
										{
											latset[i - 1][j][k] = true;
											lat[i - 1][j][k] = tmp;
										}
									}
								}
							}
						}

						if (k != 14)
						{
							for (i = 0; i < 12; i++)
							{
								for (j = 0; j < 12; j++)
								{
									for (k = 0; k < 12; k++) topo[legit][i][j][k] = lat[i][j][k];
									
								}
							}
							for (i = 0; i < 6; i++) reps[legit][i] = neigh[i];
							legit++;
						}
					}
				}
			}
		}
	}
	
	for (i = 0; i < 48; i++)
	{
		for (j = 0; j < 3; j++)
		{
			permb[i][coords[symsinv[i][j]]] = coords[j];
			if (symsinv[i][j] > 2) signb[i][coords[symsinv[i][j]]] = -1;
			else signb[i][coords[symsinv[i][j]]] = 1;
		}
	}

	for (i = 0; i < 48; i++)
	{
		for (j = 0; j < 3; j++)
		{
			perm[i][j] = permb[ginv[i]][j];
			sign[i][j] = signb[ginv[i]][j];
		}
	}

	for (l = 0; l < 187; l++)
	{
		for (i = 0; i < 12; i++)
		{
			for (j = 0; j < 12; j++)
			{
				for (k = 0; k < 12; k++)
				{
					gtopo[l][i + 12 * j + 12 * 12 * k] = topo[l][i][j][k];
				}
			}
		}
	}
	
	cudaInit(syms, gstruct, perm, sign,gtopo,ginv);

	rep0 = reps[currenttopo][0];
	rep1 = reps[currenttopo][1];
	rep2 = reps[currenttopo][2];
	rep3 = reps[currenttopo][3];
	rep4 = reps[currenttopo][4];
	rep5 = reps[currenttopo][5];

	//sf::RenderWindow window(sf::VideoMode(1920, 1080, 32), "Asteroids - Press ESC to stop", sf::Style::Titlebar | sf::Style::Close);
	sf::RenderWindow window(sf::VideoMode(1920, 1080, 32), "Asteroids - Press ESC to stop", sf::Style::Fullscreen);
	sf::Texture texture;
	sf::Sprite sprite;
	sf::Uint8* pixels = new sf::Uint8[1920 * 1080 * 4];
	sf::Vector2i winpos;

	texture.create(1920, 1080);
	window.setMouseCursorVisible(false);

	winpos = window.getPosition();
	SetCursorPos(winpos.x + 1920 / 2, winpos.y + 1080 / 2);

	while (window.isOpen())
	{
		//Sleep(1);
		sf::Event event;

		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed) window.close();


			if (!focus && event.type == sf::Event::GainedFocus) focus = true;

			if(focus)
			{
				if (event.type == sf::Event::LostFocus) focus = false;

				if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left) fireflag = true;

				if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::P) resetflag = true;

				if (event.type == sf::Event::MouseMoved)
				{
					POINT p;
					GetCursorPos(&p);
					winpos = window.getPosition();
					centralx = winpos.x + 1920 / 2;
					centraly = winpos.y + 1080 / 2;
					SetCursorPos(centralx, centraly);

					mousx = p.x - centralx;
					mousy = p.y - centraly;

					anglex = 0.002 * mousx;
					angley = 0.002 * mousy;

					if (mousx > 0)
					{
						for (j = 0; j < 3; j++)
						{
							newx[0][j] = x[0][j] * cos(anglex) + sin(anglex) * x[1][j];
							newx[1][j] = x[1][j] * cos(anglex) - sin(anglex) * x[0][j];
							x[0][j] = newx[0][j];
							x[1][j] = newx[1][j];
						}
					}
					else if (mousx < 0)
					{
						anglex = -anglex;

						for (j = 0; j < 3; j++)
						{
							newx[0][j] = x[0][j] * cos(anglex) - sin(anglex) * x[1][j];
							newx[1][j] = x[1][j] * cos(anglex) + sin(anglex) * x[0][j];
							x[0][j] = newx[0][j];
							x[1][j] = newx[1][j];
						}

					}

					if (mousy < 0)
					{
						angley = -angley;
						for (j = 0; j < 3; j++)
						{
							newx[0][j] = x[0][j] * cos(angley) + sin(angley) * x[2][j];
							newx[2][j] = x[2][j] * cos(angley) - sin(angley) * x[0][j];
							x[0][j] = newx[0][j];
							x[2][j] = newx[2][j];
						}

					}
					else if (mousy > 0)
					{
						for (j = 0; j < 3; j++)
						{
							newx[0][j] = x[0][j] * cos(angley) - sin(angley) * x[2][j];
							newx[2][j] = x[2][j] * cos(angley) + sin(angley) * x[0][j];
							x[0][j] = newx[0][j];
							x[2][j] = newx[2][j];
						}
					}


				}

				if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Z)
				{
					currenttopo++;
					currenttopo %= 187;

					rep0 = reps[currenttopo][0];
					rep1 = reps[currenttopo][1];
					rep2 = reps[currenttopo][2];
					rep3 = reps[currenttopo][3];
					rep4 = reps[currenttopo][4];
					rep5 = reps[currenttopo][5];

					for (i = 0; i < 6; i++) walls[i] = false;
					for (i = 1; i < 6; i++) walltmp[i] = true;

					wp[0][1] = syms[rep0][5];
					walltmp[wp[0][1]] = false;
					walltmpidx = 1;

					for (i = 1; i < 6; i++)
					{
						if (walltmp[i])
						{
							wp[walltmpidx][0] = i;
							wp[walltmpidx][1] = syms[reps[currenttopo][i]][opp[i]];
							walltmp[wp[walltmpidx][1]] = false;
							walltmpidx++;
						}
					}
				}
				if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::X)
				{
					currenttopo += 10;
					currenttopo %= 187;

					rep0 = reps[currenttopo][0];
					rep1 = reps[currenttopo][1];
					rep2 = reps[currenttopo][2];
					rep3 = reps[currenttopo][3];
					rep4 = reps[currenttopo][4];
					rep5 = reps[currenttopo][5];

					for (i = 0; i < 6; i++) walls[i] = false;
					for (i = 1; i < 6; i++) walltmp[i] = true;

					wp[0][1] = syms[rep0][5];
					walltmp[wp[0][1]] = false;
					walltmpidx = 1;

					for (i = 1; i < 6; i++)
					{
						if (walltmp[i])
						{
							wp[walltmpidx][0] = i;
							wp[walltmpidx][1] = syms[reps[currenttopo][i]][opp[i]];
							walltmp[wp[walltmpidx][1]] = false;
							walltmpidx++;
						}
					}
				}
				if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::C)
				{
					currenttopo += 100;
					currenttopo %= 187;

					rep0 = reps[currenttopo][0];
					rep1 = reps[currenttopo][1];
					rep2 = reps[currenttopo][2];
					rep3 = reps[currenttopo][3];
					rep4 = reps[currenttopo][4];
					rep5 = reps[currenttopo][5];

					for (i = 0; i < 6; i++) walls[i] = false;
					for (i = 1; i < 6; i++) walltmp[i] = true;

					wp[0][1] = syms[rep0][5];
					walltmp[wp[0][1]] = false;
					walltmpidx = 1;

					for (i = 1; i < 6; i++)
					{
						if (walltmp[i])
						{
							wp[walltmpidx][0] = i;
							wp[walltmpidx][1] = syms[reps[currenttopo][i]][opp[i]];
							walltmp[wp[walltmpidx][1]] = false;
							walltmpidx++;
						}
					}
				}

				if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::F)
				{
					walls[wp[0][0]] = !walls[wp[0][0]];
					walls[wp[0][1]] = !walls[wp[0][1]];
				}
				if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::G)
				{
					walls[wp[1][0]] = !walls[wp[1][0]];
					walls[wp[1][1]] = !walls[wp[1][1]];

				}
				if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::H)
				{
					walls[wp[2][0]] = !walls[wp[2][0]];
					walls[wp[2][1]] = !walls[wp[2][1]];
				}
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) window.close();
		
		
		if (focus)
		{
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::M))  sdp = true;
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::N))  sdm = true;
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::K))  fovm = true;
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::L))  fovp = true;
			
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::W))
			{
				cv[0] += speed*x[0][0];
				cv[1] += speed * x[0][1];
				cv[2] += speed * x[0][2];

			}

			if (fovp) dist += 0.05;
			else if (fovm)
			{
				if (dist > 0.05) dist -= 0.05;
			}

			pos0 += cv[0];
			pos1 += cv[1];
			pos2 += cv[2];

			pos0 -= 0.5;
			pos1 -= 0.5;
			pos2 -= 0.5;

			if (walls[0])
			{
				tmpd = pos2 + csize - 0.5;
				if (tmpd > 0)
				{
					pos2 -= tmpd;
					cv[2] *= -1;
				}
			}
			if (walls[1])
			{
				tmpd = pos0 + csize - 0.5;
				if (tmpd > 0)
				{
					pos0 -= tmpd;
					cv[0] *= -1;
				}
			}
			if (walls[2])
			{
				tmpd = pos1 + csize - 0.5;
				if (tmpd > 0)
				{
					pos1 -= tmpd;
					cv[1] *= -1;
				}
			}
			if (walls[5])
			{
				tmpd = pos2 - csize + 0.5;
				if (tmpd < 0)
				{
					pos2 -= tmpd;
					cv[2] *= -1;
				}
			}
			if (walls[3])
			{
				tmpd = pos0 - csize+ 0.5;
				if (tmpd < 0)
				{
					pos0 -= tmpd;
					cv[0] *= -1;
				}
			}
			if (walls[4])
			{
				tmpd = pos1 - csize + 0.5;
				if (tmpd < 0)
				{
					pos1 -= tmpd;
					cv[1] *= -1;
				}
			}

			pos0 += 0.5;
			pos1 += 0.5;
			pos2 += 0.5;

			ppos0 = pos0;
			ppos1 = pos1;
			ppos2 = pos2;

			if (pos0 < 0) ppos0--;
			if (pos1 < 0) ppos1--;
			if (pos2 < 0) ppos2--;

			if (ppos0 != 0 || ppos1 != 0 || ppos2 != 0)
			{


				ppos0 %= 12;
				ppos1 %= 12;
				ppos2 %= 12;
				if (ppos0 < 0)ppos0 += 12;
				if (ppos1 < 0)ppos1 += 12;
				if (ppos2 < 0)ppos2 += 12;


				newsym = topo[currenttopo][ppos2][ppos1][ppos0];

				pos0 = fmod(pos0, 1.0);
				if (pos0 < 0) pos0++;
				pos0 -= 0.5;
				pos1 = fmod(pos1, 1.0);
				if (pos1 < 0) pos1++;
				pos1 -= 0.5;
				pos2 = fmod(pos2, 1.0);
				if (pos2 < 0) pos2++;
				pos2 -= 0.5;

				postmp[0] = pos0;
				postmp[1] = pos1;
				postmp[2] = pos2;

				pos0 = sign[newsym][0] * postmp[perm[newsym][0]];
				pos1 = sign[newsym][1] * postmp[perm[newsym][1]];
				pos2 = sign[newsym][2] * postmp[perm[newsym][2]];

				pos0 += 0.5;
				pos1 += 0.5;
				pos2 += 0.5;

				for (i = 0; i < 3; i++) for (j = 0; j < 3; j++) newx[i][j] = sign[newsym][j] * x[i][perm[newsym][j]];

				for (i = 0; i < 3; i++) for (j = 0; j < 3; j++) x[i][j] = newx[i][j];

				for (j = 0; j < 3; j++) newcv[j] = sign[newsym][j] * cv[perm[newsym][j]];

				for (j = 0; j < 3; j++) cv[j] = newcv[j];

			}
			if (resetflag)
			{
				pos0 = 0.5; pos1 = 0.5; pos2 = 0.01;
				dist = 2;
				cv[0] = 0;
				cv[1] = 0;
				cv[2] = 0;
			}

			vec0 = dist * x[0][0] + multy * x[1][0] + multz * x[2][0];
			vec1 = dist * x[0][1] + multy * x[1][1] + multz * x[2][1];
			vec2 = dist * x[0][2] + multy * x[1][2] + multz * x[2][2];

			addy0 = sqsz * x[1][0];
			addy1 = sqsz * x[1][1];
			addy2 = sqsz * x[1][2];

			addz0 = -sqsz * x[2][0];
			addz1 = -sqsz * x[2][1];
			addz2 = -sqsz * x[2][2];

			

			
			
			det = x[0][0] * x[1][1] * x[2][2] + x[0][2] * x[1][0] * x[2][1] + x[2][0] * x[0][1] * x[1][2] - x[2][0] * x[1][1] * x[0][2] - x[1][0] * x[0][1] * x[2][2] - x[0][0] * x[2][1] * x[1][2];
			m0 = (x[1][1]*x[2][2]-x[2][1]*x[1][2]) / det;
			m1 = (x[2][0] * x[1][2] - x[1][0] * x[2][2]) / det;
			m2 = (x[1][0] * x[2][1] - x[2][0] * x[1][1]) / det;
			m3 = (x[2][1] * x[0][2] - x[0][1] * x[2][2]) / det;
			m4 = (x[0][0] * x[2][2] - x[2][0] * x[0][2]) / det;
			m5 = (x[2][0] * x[0][1] - x[0][0] * x[2][1]) / det;
			m6 = (x[0][1] * x[1][2] - x[1][1] * x[0][2]) / det;
			m7 = (x[1][0] * x[0][2] - x[0][0] * x[1][2]) / det;
			m8 = (x[0][0] * x[1][1] - x[1][0] * x[0][1]) / det;
			
			cudathingy(pixels, pos0, pos1, pos2, vec0, vec1, vec2, addy0, addy1, addy2, addz0, addz1, addz2,rep0,rep1,rep2,rep3,rep4,rep5,currenttopo,fireflag, x[0][0],x[0][1],x[0][2],nbframe,resetflag,sdp,sdm, fovp, fovm,cv[0],cv[1],cv[2],walls[0],walls[1],walls[2],walls[3],walls[4],walls[5],m0,m1,m2,m3,m4,m5,m6,m7,m8);

			for (i = 537; i < 544; i++)
			{
					pixels[4 * (1920 * i + 960)] = 255;
					pixels[4 * (1920 * i + 960) + 1] = 0;
					pixels[4 * (1920 * i + 960) + 2] = 255;
			}

			for (j = 957; j < 964; j++)
			{
				pixels[4 * (1920 * 540 + j)] = 255;
				pixels[4 * (1920 * 540 + j) + 1] = 0;
				pixels[4 * (1920 * 540 + j) + 2] = 255;
			}

			nbframe++;
			fireflag = false;
			resetflag = false;
			sdp = false;
			sdm = false;
			fovp = false;
			fovm = false;
			texture.update(pixels);
			sprite.setTexture(texture);
			window.draw(sprite);
			window.display();
		}
		else Sleep(10);
	}


	cudaExit();
	return 0;
}