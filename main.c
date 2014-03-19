#include <OpenCV/OpenCV.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sqlite3.h>

#define LOG_ERROR_SEVERE 3
#define LOG_ERROR 2
#define LOG_MSG 1

CvCapture *capture;
CvMemStorage *mem, *longterm;
IplImage *frame, *tmp, *tmp_disp;
IplImage *luv, *luv_disp;
IplImage *gray, *gray_disp;

sqlite3 *db;
int rc;

typedef struct {
	CvPoint pos;
	CvPoint vel;
	int age;
	int type;
	int age_t0;
	int age_t1;
}hand;

int conf_approx_poly=10;
int conf_min_edge_len=30;
int conf_min_size=1000;
int conf_luv_thresh_u_low=85;
int conf_luv_thresh_u_up=95;
int conf_luv_thresh_v_low=100;
int conf_luv_thresh_v_up=130;
int conf_bin_thresh=200;
int conf_border_left=5;
int conf_border_top=5;
int conf_border_right=5;
int conf_border_bottom=5;
int conf_border_dist=10;
char *conf_sql_db="sqldb";

int conf_hand_age=3;
int conf_hand_max_vel=50;

int f_counter;
char msg[128];

char *config;
int verbose;

void init( char *f, char *c, char *m, char *cam);
void terminate();
void p(int n, int l);
void save_config(char *config);
void load_config(char *config);

void process(IplImage *in);
void luv_thresh(IplImage *in, IplImage *out, int u_low, int u_up, int v_low, int v_up);
CvPoint* detect_finger(CvPoint* p1, CvPoint* p2, CvPoint* p3);
void remove_short(CvSeq* seq);
CvSeq* detect_fingers(CvSeq* seq);
CvPoint get_center(CvSeq* seq);
void remove_edge_points(CvSeq* seq, int bl, int bt, int br, int bb, int bd);
void draw_contour(IplImage* img, CvSeq* s);
//void draw_hand(IplImage *img, CvSeq *seq);
void draw_hands(IplImage *img);

void hands_age();
void hands_die();
CvPoint* hands_process(CvPoint *p, int t);
void spring(hand* h, CvPoint *p, int t);

void cb_min(int pos);
void cb_bl(int pos);
void cb_bt(int pos);
void cb_br(int pos);
void cb_bb(int pos);
void cb_bd(int pos);
void cb_luv(int pos);

double dist(CvPoint* p1, CvPoint* p2);

CvSeq *hands;

const char* ARG_CAM="--cam=";
const char* ARG_CONFIG = "--config=";
const char* ARG_MODE="--mode=";


int main(int argc, char** argv){
	int i, key;
	char *avi, *mode, *cam;
	p(sprintf(msg, "Hand Recognition"), LOG_MSG);
	avi="/film1.avi";
	config="/config.ini";
	mode = "2";
	cam="";
	for(i=1;i<argc;i++){
		if(strncmp(argv[i], ARG_CAM, strlen(ARG_CAM))==0)
			cam=argv[i]+strlen(ARG_CAM);
		else if(strncmp(argv[i], ARG_CONFIG, strlen(ARG_CONFIG))==0)
			config=argv[i]+strlen(ARG_CONFIG);
		else if(strncmp(argv[i], ARG_MODE, strlen(ARG_MODE))==0)
			mode=argv[i]+strlen(ARG_MODE);
		else if(argv[i][0]=='-')
			p(sprintf(msg, "what the heck is %s?\n", argv[i]), LOG_ERROR);
		else
			avi=argv[i];
	}
	
	
	init(avi, config, mode, cam);
	key=0;
	if(capture)
        for(;key!='q';key=cvWaitKey(1)){
			frame = cvQueryFrame(capture);
            if( !frame ){
				p(sprintf(msg, "could not query frame"), LOG_ERROR_SEVERE);
			}
                
			hands_age();
			
			f_counter++;
			
			process(frame);
			
			hands_die();
			
			if(verbose>0){
				draw_hands(frame);
				cvRectangle(frame, cvPoint(conf_border_left, conf_border_top), cvPoint(cvGetSize(frame).width-conf_border_right, cvGetSize(frame).height-conf_border_bottom), CV_RGB(255, 0, 0), 1, 8, 0);
				cvRectangle(frame, cvPoint(conf_border_left+conf_border_dist, conf_border_top+conf_border_dist), cvPoint(cvGetSize(frame).width-conf_border_right-conf_border_dist, cvGetSize(frame).height-conf_border_bottom-conf_border_dist), CV_RGB(0, 0, 255), 1, 8, 0);
				if(verbose>1){
					cvShowImage("luv", luv_disp);
					cvShowImage("gray", gray_disp);
				}
				cvShowImage("recog", frame);
			}
			//usleep(20000);
			//usleep(50000);
			if(key==' ')
				cvWaitKey(0);
		}
	terminate();
	return 0;
}

void process(IplImage *img){
	
	cvResize(img, luv_disp, CV_INTER_LINEAR);
	luv_thresh(luv_disp, luv_disp, conf_luv_thresh_u_low, conf_luv_thresh_u_up, conf_luv_thresh_v_low, conf_luv_thresh_v_up);
	
	cvCvtColor(luv_disp, gray_disp, CV_BGR2GRAY);
	cvResize(gray_disp, gray, CV_INTER_NN);

	mem = cvCreateMemStorage(0);
	
	CvSeq* contour = 0;
	cvFindContours(gray, mem, &contour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));
	if(contour)
		for(;contour!=0;contour=contour->h_next){
			int i;
			float f = cvContourArea(contour, CV_WHOLE_SEQ);
			if(fabs(f)>conf_min_size ){
				CvSeq* c = cvApproxPoly(contour, sizeof(CvContour), mem, CV_POLY_APPROX_DP, conf_approx_poly, 0);
				
				CvSeq* sharp = detect_fingers(c);
				draw_contour(frame, c);
				remove_edge_points(sharp, conf_border_left, conf_border_top, cvGetSize(img).width-conf_border_right, cvGetSize(img).height-conf_border_bottom, conf_border_dist);
				for( i=0; i<sharp->total; ++i ) { 
					CvPoint* p = (CvPoint*)cvGetSeqElem ( sharp, i ); 
					cvCircle(frame, *p, 2, cvScalarAll(255), 2, 8, 0);
				}
				if(sharp->total==1){
					hands_process((CvPoint*)cvGetSeqElem ( sharp, 0 ), 0);
				}else if(sharp->total>1 ){
					CvPoint t = get_center(sharp);
					hands_process(&t, 1);
				}
				
			}
		}
		
	cvReleaseMemStorage(&mem);
	//  */
	
}
void hands_age(){
	int i;
	hand *h;
	for(i=0;i<hands->total;i++){
		h = (hand*)cvGetSeqElem(hands, i);
		h->age=h->age+1;
	}
}
void hands_die(){
	int i;
	hand *h;
	for(i=0;i<hands->total;i++){
		h = (hand*)cvGetSeqElem(hands, i);
		if(h->age>conf_hand_age){
			cvSeqRemove(hands, i);
			--i;
		}
	}
}
CvPoint* hands_process(CvPoint *p, int t){
	int i;
	hand *h;
	int index=-1;
	for(i=0;i<hands->total;i++){
		h = (hand*)cvGetSeqElem(hands, i);
		if(dist(p, &(h->pos))<conf_hand_max_vel){
			index=i;
			break;
		}
	}
	if(index<0){
		h = (hand*)malloc(sizeof(hand));
		h->age = 0;
		h->type = t;
		h->pos = cvPoint(p->x, p->y);
		h->vel = cvPoint(0, 0);
		cvSeqPush(hands, h);
	}else{
		h = (hand*)cvGetSeqElem(hands, index);
		
			spring(h, p, t);
	
		
		if(t==0){
			h->age_t0++;
			h->age_t1--;
			if(h->age_t0>h->age_t1)
				h->type=t;
			
		}else{
			h->age_t1++;
			h->age_t0--;
			if(h->age_t1>h->age_t0)
				h->type=t;
		}
		h->age_t0 = h->age_t0 < 0 ? 0 : h->age_t0;
		h->age_t1 = h->age_t1 < 0 ? 0 : h->age_t1;
		h->age_t0 = h->age_t0 > conf_hand_age ? conf_hand_age : h->age_t0;
		h->age_t1 = h->age_t1 > conf_hand_age ? conf_hand_age : h->age_t1;
		h->age=0;
	}
	return &(h->pos);
}
void spring(hand *h, CvPoint *p, int t){
	float D, friction;
	D=0.7f;
	friction=0.65f;
	if(t>0)
		friction=0.35f;
	int dx, dy;
	dx = h->pos.x - p->x;
	dy = h->pos.y - p->y;
	h->vel.x += (p->x - h->pos.x) * D;
	h->vel.y += (p->y - h->pos.y) * D;
	h->vel.x *= friction;
	h->vel.y *= friction;
	h->pos.x += h->vel.x;
	h->pos.y += h->vel.y;
}
void luv_thresh(IplImage *in, IplImage *out, int u_low, int u_up, int v_low, int v_up){
	int x, y;
	
	cvCvtColor(in, out, CV_RGB2Luv);
	for(y=0; y<out->height;y++){
		uchar* ptr = (uchar*)(out->imageData + y * out->widthStep); 
		for(x=0;x<out->width;x++){
			if(ptr[3*x+1]<u_low||ptr[3*x+1]>u_up||ptr[3*x+2]<v_low||ptr[3*x+2]>v_up){
				ptr[3*x+1]=0;
				ptr[3*x+2]=0;
				ptr[3*x]=0;
			}
		}
	}
}

void draw_hands(IplImage *img){
	int i;
	hand *h;
	for(i=0;i<hands->total;i++){
		h = (hand*)cvGetSeqElem(hands, i);
		cvCircle(img, h->pos, 30, CV_RGB(h->type*255, (-1)*(h->type-1)*255,0), 2, 8, 0);
	}
}
CvPoint* detect_finger(CvPoint* p1, CvPoint* p2, CvPoint* p3){
	
	CvPoint* result;
	
	result =0;
	if(p1==0||p2==0||p3==0)
		return 0;
	double a = dist(p2, p3);
	double b = dist(p3, p1);
	double c = dist(p1, p2);
	
	double angle = acos((a*a+c*c-b*b)/(2*a*c));
	if(angle<0.7)
		result=p2;
	
	return result;
}
void remove_short(CvSeq* seq){
	
	int i;
	CvPoint *p1,*p2;
	for( i=0; i<seq->total-1; ++i ) { 
		p1 = (CvPoint*)cvGetSeqElem ( seq, i ); 
		p2 = (CvPoint*)cvGetSeqElem ( seq, i +1);
		if(dist(p1,p2)<conf_min_edge_len){
			cvSeqRemove(seq, i+1);
			--i;
		}
	}
	if(dist(p2,(CvPoint*)cvGetSeqElem ( seq, 0 ))<conf_min_edge_len){
		cvSeqRemove(seq, 0);
	}
}
CvSeq* detect_fingers(CvSeq* seq){
	
	int i;
	CvSeq* sharp = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), mem);
	
	remove_short(seq);
	
	CvPoint *x, *p1, *p2, *p3;
	for( i=0; i<(seq->total-2); ++i ) { 
		p1 = (CvPoint*)cvGetSeqElem ( seq, i ); 
		p2 = (CvPoint*)cvGetSeqElem ( seq, i +1);
		p3 = (CvPoint*)cvGetSeqElem ( seq, i +2);
		x=detect_finger(p1,p2,p3);
		if(x!=0){
			cvSeqPush(sharp, p2);
			++i;
		}
	} 
	x=detect_finger(p2, p3, (CvPoint*)cvGetSeqElem ( seq, 0 ));
	if(x!=0)
		cvSeqPush(sharp, p3);
	x=detect_finger(p3, (CvPoint*)cvGetSeqElem ( seq, 0 ), (CvPoint*)cvGetSeqElem ( seq, 1 ));
	if(x!=0)
		cvSeqPush(sharp, (CvPoint*)cvGetSeqElem ( seq, 0 ));
	return sharp;
}
CvPoint get_center(CvSeq* seq){
	
	CvPoint c = cvPoint(0,0);
	int i;
	for( i=0; i<seq->total; ++i ) { 
		CvPoint* p = (CvPoint*)cvGetSeqElem ( seq, i ); 
		c.x += p->x;
		c.y += p->y;
	}
	c.x /= seq->total;
	c.y /= seq->total;
	return c;
}
void remove_edge_points(CvSeq* seq, int bl, int bt, int br, int bb, int bd){
	
	int i;
	
	for( i=0; i<seq->total; ++i ) { 
		CvPoint* p = (CvPoint*)cvGetSeqElem ( seq, i );
		if(p->x < bl+bd || p->x > br-bd || p->y < bt+bd || p->y > bb-bd){
			//printf("%d-%d-%d, %d-%d-%d, %d\n", bl, p->x, br, bt, p->y, bb, bd);
			cvSeqRemove(seq, i);
			--i;
		}
	}
}
void draw_contour(IplImage* img, CvSeq* s){
	int i;
	CvPoint *p1, *p2, *start;
	p2=0;
	start = (CvPoint*)cvGetSeqElem ( s, 0 );
	for( i=0; i<s->total-1; ++i ) { 
		p1 = (CvPoint*)cvGetSeqElem ( s, i );
		p2 = (CvPoint*)cvGetSeqElem ( s, i +1);
		cvCircle(img, *p1, 0, cvScalarAll(255), 3, 8, 0);
		cvLine(img, *p1, *p2, CV_RGB(64,64,64), 1, 8, 0);	
	}
	if(p2!=0){
		cvCircle(img, *p2, 0, cvScalarAll(255), 3, 8, 0);
		cvLine(img, *p2, *start, CV_RGB(64,64,64), 1, 8, 0);
	}
}

double dist(CvPoint* p1, CvPoint* p2){
	
	int dx = p1->x - p2->x;
	int dy = p1->y - p2->y;
	return sqrt(dx*dx + dy*dy);
}

void cb_min(int pos){
	cvCopy(frame, tmp, 0);
	cvRectangle(tmp, cvPoint(0,0), cvPoint(pos, 10), CV_RGB(0, 255, 0), CV_FILLED, 8, 0);
	if(verbose>1)
		cvShowImage("recog", tmp);
}
void cb_bl(int pos){
	cvCopy(frame, tmp, 0);
	cvRectangle(tmp, cvPoint(0,0), cvPoint(pos, cvGetSize(frame).height), CV_RGB(0, 255, 0), CV_FILLED, 8, 0);
	if(verbose>1)
		cvShowImage("recog", tmp);
}
void cb_bt(int pos){
	cvCopy(frame, tmp, 0);
	cvRectangle(tmp, cvPoint(0,0), cvPoint(cvGetSize(frame).width, pos), CV_RGB(0, 255, 0), CV_FILLED, 8, 0);
	if(verbose>1)
		cvShowImage("recog", tmp);
}
void cb_br(int pos){
	cvCopy(frame, tmp, 0);
	CvSize sz = cvGetSize(frame);
	cvRectangle(tmp, cvPoint(sz.width-pos,0), cvPoint(sz.width, sz.height), CV_RGB(0, 255, 0), CV_FILLED, 8, 0);
	if(verbose>1)
		cvShowImage("recog", tmp);
}
void cb_bb(int pos){
	cvCopy(frame, tmp, 0);
	CvSize sz = cvGetSize(frame);
	cvRectangle(tmp, cvPoint(0,sz.height-pos), cvPoint(sz.width, sz.height), CV_RGB(0, 255, 0), CV_FILLED, 8, 0);
	if(verbose>1)
		cvShowImage("recog", tmp);
}
void cb_bd(int pos){
	cvCopy(frame, tmp, 0);
	cvRectangle(tmp, cvPoint(conf_border_left+conf_border_dist, conf_border_top+conf_border_dist), cvPoint(cvGetSize(frame).width-conf_border_right-conf_border_dist, cvGetSize(frame).height-conf_border_bottom-conf_border_dist), CV_RGB(0, 0, 255), 1, 8, 0);
	if(verbose>1)
		cvShowImage("recog", tmp);
}
void cb_luv(int pos){
	cvCopy(frame, tmp, 0);
	cvResize(tmp, luv_disp, CV_INTER_LINEAR);
	luv_thresh(luv_disp, luv_disp, conf_luv_thresh_u_low, conf_luv_thresh_u_up, conf_luv_thresh_v_low, conf_luv_thresh_v_up);
	
	if(verbose>1)
		cvShowImage("luv", luv_disp);
}
void init( char* f, char *c, char *m, char *cam){
	f_counter=0;
	load_config(c);

	//p(sprintf(msg, "Initialize with f=%s", file), LOG_MSG);
	//printf("hoi");
	
	
	if(cam==0|| cam==""){
		capture = cvCaptureFromAVI(f);
	}else{
		int cn = *cam - '0';
		capture = cvCaptureFromCAM(cn);
	}
	verbose = (*m-'0');
	if(verbose>0){
		if(verbose>1){
			cvNamedWindow("luv", 1);
			cvMoveWindow("luv", 0, 620);
			cvNamedWindow("gray", 1);
			cvMoveWindow("gray", 320, 620);
		
			cvNamedWindow("settings", 1);
			cvMoveWindow("settings", 720, 44);
	
			//cvNamedWindow("tmp", 1);
	
			cvCreateTrackbar("approx_poly", "settings", &conf_approx_poly, 30, 0);
			cvCreateTrackbar("min_edge_len", "settings", &conf_min_edge_len, 100, cb_min);
			cvCreateTrackbar("min_size", "settings", &conf_min_size, 10000, 0);
			cvCreateTrackbar("luv_thresh_u_low", "settings", &conf_luv_thresh_u_low, 255, cb_luv);
			cvCreateTrackbar("luv_thresh_u_up", "settings", &conf_luv_thresh_u_up, 255, cb_luv);
			cvCreateTrackbar("luv_thresh_v_low", "settings", &conf_luv_thresh_v_low, 255, cb_luv);
			cvCreateTrackbar("luv_thresh_v_up", "settings", &conf_luv_thresh_v_up, 255, cb_luv);
			//cvCreateTrackbar("bin_thresh", "settings", &conf_bin_thresh, 255, 0);
			cvCreateTrackbar("border_left", "settings", &conf_border_left, 150, cb_bl);
			cvCreateTrackbar("border_top", "settings", &conf_border_top, 150, cb_bt);
			cvCreateTrackbar("border_right", "settings", &conf_border_right, 150, cb_br);
			cvCreateTrackbar("border_bottom", "settings", &conf_border_bottom, 150, cb_bb);
			cvCreateTrackbar("border_dist", "settings", &conf_border_dist, 100, cb_bd);
			cvCreateTrackbar("hand_age", "settings", &conf_hand_age, 20, 0);
			cvCreateTrackbar("hand_max_vel", "settings", &conf_hand_max_vel, 200, cb_min);
		}
		cvNamedWindow("recog", 1);
		cvMoveWindow("recog", 0, 44);
	}
	longterm = cvCreateMemStorage(0);
	hands = cvCreateSeq(CV_SEQ_ELTYPE_GENERIC, sizeof(CvSeq), sizeof(hand), longterm);
	
	frame = cvQueryFrame(capture);
	if(!frame){
		p(sprintf(msg, "Could not grab frame"), LOG_ERROR_SEVERE);
	}
	
	CvSize sz = cvGetSize(frame);
	CvSize small = cvSize(sz.width/4, sz.height/4);
	
	luv_disp = cvCreateImage(small, IPL_DEPTH_8U, 3);
	luv = cvCreateImage(sz, IPL_DEPTH_8U, 3);
	gray = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	gray_disp = cvCreateImage(small, IPL_DEPTH_8U, 1);
	tmp = cvCreateImage(sz, IPL_DEPTH_8U, 3);
	//tmp_disp = cvCreateImage(small, IPL_DEPTH_8U, 3);
	// 
	rc = sqlite3_open(conf_sql_db, &db);
	  if( rc ){
	    fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
	    sqlite3_close(db);
	    exit(1);
	  }
}
void terminate(){
	//cvReleaseImage(&frame);
	//cvReleaseImage(&luv_disp);
	sqlite3_close(db);
	cvReleaseMemStorage(&longterm);
	save_config(config);
	cvDestroyAllWindows();
	cvReleaseCapture(&capture);
}
void p(int n, int l){

	if(l>=LOG_ERROR){
		printf("(%d) ERROR: %s\n", f_counter, msg);
	}
	if(l>=LOG_ERROR_SEVERE){
		printf("       exiting.\n");
		terminate();
		exit(0);
	}
	if(l==LOG_MSG){
		printf("(%d) INFO: %s\n", f_counter, msg);
	}
}
void load_config(char *c){
	FILE * pFile;
	char str[50];
	int val;
	pFile = fopen (c,"r");
	if (pFile!=0){
		while (!feof(pFile)) {
			fscanf(pFile, "%s %d", &str, &val);
			if(strcmp(str, "border_left")==0)
				conf_border_left=val;
			if(strcmp(str, "border_right")==0)
				conf_border_right=val;
			if(strcmp(str, "border_top")==0)
				conf_border_top=val;
			if(strcmp(str, "border_bottom")==0)
				conf_border_bottom=val;
			if(strcmp(str, "border_dist")==0)
				conf_border_dist=val;
			if(strcmp(str, "luv_ul")==0)
				conf_luv_thresh_u_low=val;
			if(strcmp(str, "luv_uh")==0)
				conf_luv_thresh_u_up=val;
			if(strcmp(str, "luv_vl")==0)
				conf_luv_thresh_v_low=val;
			if(strcmp(str, "luv_vh")==0)
				conf_luv_thresh_v_up=val;
			if(strcmp(str, "min_edge_len")==0)
				conf_min_edge_len=val;
			if(strcmp(str, "min_size")==0)
				conf_min_size=val;
			if(strcmp(str, "approx_poly")==0)
				conf_approx_poly=val;
			if(strcmp(str, "hand_age")==0)
				conf_hand_age=val;
			if(strcmp(str, "hand_max_vel")==0)
				conf_hand_max_vel=val;
		}
		fclose (pFile);
		
	}
}
void save_config(char *c){
	FILE * pFile;
	//char str[50];
	//int val;
	pFile = fopen (c,"w");
	if (pFile!=0){
		fprintf(pFile, "%s %d\n", "border_left", conf_border_left);
		fprintf(pFile, "%s %d\n", "border_right", conf_border_right);
		fprintf(pFile, "%s %d\n", "border_top", conf_border_top);
		fprintf(pFile, "%s %d\n", "border_bottom", conf_border_bottom);
		fprintf(pFile, "%s %d\n", "border_dist", conf_border_dist);
		fprintf(pFile, "%s %d\n", "luv_vl", conf_luv_thresh_v_low);
		fprintf(pFile, "%s %d\n", "luv_vh", conf_luv_thresh_v_up);
		fprintf(pFile, "%s %d\n", "luv_ul", conf_luv_thresh_u_low);
		fprintf(pFile, "%s %d\n", "luv_uh", conf_luv_thresh_u_up);
		fprintf(pFile, "%s %d\n", "min_edge_len", conf_min_edge_len);
		fprintf(pFile, "%s %d\n", "min_size", conf_min_size);
		fprintf(pFile, "%s %d\n", "approx_poly", conf_approx_poly);
		fprintf(pFile, "%s %d\n", "hand_age", conf_hand_age);
		fprintf(pFile, "%s %d\n", "hand_max_vel", conf_hand_max_vel);
		fclose (pFile);
		
	}
}
