#define _USE_MATH_DEFINES

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>  
#include <vector>  
#include <cmath>
#include <numbers>
#include <iomanip>
#include <chrono>

using namespace std;

struct Point
{
    float x;
    float y;
};

float min_x = 999999999;
float max_x = -1;
float min_y = 999999999;
float max_y = -1;

float min_shift = 0.00001;

float radius = 0.1f;
int n = 200;
int m = 0;

float cluster_max_distance = 0.01;

vector<Point> points;
vector<Point> r_points;

std::chrono::high_resolution_clock::time_point t1;
std::chrono::high_resolution_clock::time_point t2;

void start_timer() {
    std::cout << "Timer started" << endl;
    t1 = std::chrono::high_resolution_clock::now();
}

void end_timer() {
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Timer ended" << endl;
}
void print_elapsed_time() {
    std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " milliseconds\n" << endl;
}
void read_data() {
    fstream newfile;
    newfile.open("s1.txt", ios::in); //open a file to perform read operation using file object
    std::ifstream file("s1.txt");
    if (!file.is_open()) {
        std::cout << "Failed to open file: " << std::endl;
    }

    double x, y;
    while (file >> x >> y) {
        Point point = { x, y };
        if (x < min_x)
            min_x = x;
        if (y < min_y)
            min_y = y;
        if (x > max_x)
            max_x = x;
        if (y > max_y)
            max_y = y;
        points.push_back(point);
        m++;
    }
    
}

float lerp(float v, float a, float b) {
        return (v - a) / (b - a);
    }

void normalize_data() {
    for (int i = 0; i < points.size(); i++) {

        //cout << points[i].x << " " << points[i].y << endl;
        points[i].x = lerp(points[i].x, min_x, max_x);
        points[i].y = lerp(points[i].y, min_y, max_y);
        //cout << points[i].x << " " << points[i].y << endl;
    }
}



float calculate_distance(Point a, Point b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

Point calculate_shift(Point p1) {
    bool done = false;

    while (!done) {
        float sum1_x = 0.f;
        float sum1_y = 0.f;
        float sum2 = 0.f;

#pragma omp parallel for
        for (int i = 0; i < m; i++) {
            Point p2 = points[i];
            float distance = calculate_distance(p1, p2);
            if (distance > radius) {
                continue;
            }
            //sum1_x += 1 / (sqrt(2 * M_PI) * radius) * exp((- 1) * (pow(distance, 2) / (2 * pow(radius, 2)))) * p2.x;
            //sum1_y += 1 / (sqrt(2 * M_PI) * radius) * exp((- 1) * (pow(distance, 2) / (2 * pow(radius, 2)))) * p2.y;
            //sum2 += 1 / (sqrt(2 * M_PI) * radius) * exp((-1) * (pow(distance, 2) / (2 * pow(radius, 2))));

            sum1_x += (1 / pow((sqrt(2 * M_PI)) * 1, 2)) * exp((-1) * (pow(distance, 2) / (2 * pow(1, 2)))) * p2.x;
            sum1_y += (1 / pow((sqrt(2 * M_PI)) * 1, 2)) * exp((-1) * (pow(distance, 2) / (2 * pow(1, 2)))) * p2.y;
            sum2 +=   (1 / pow((sqrt(2 * M_PI)) * 1, 2)) * exp((-1) * (pow(distance, 2) / (2 * pow(1, 2))));
            //sum1_x += p2.x;
            //sum1_y += p2.y;
            //sum2 += 1.f;
        }
        float x_shifted = sum1_x / sum2;
        float y_shifted = sum1_y / sum2;
        
        Point p_shifted;
        p_shifted.x = x_shifted;
        p_shifted.y = y_shifted;

        float tmp_distance = calculate_distance(p1, p_shifted);
        
        if (tmp_distance < min_shift)
            done = true;

        p1.x = x_shifted;
        p1.y = y_shifted;

    }

    return p1;
}

void mean_shift() {
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        Point p = calculate_shift(r_points[i]);
        r_points[i] = p;
    }
}

void calculate_clusters() {
    int count = 1;
    vector<Point> cluster_points;
    cluster_points.push_back(r_points[0]);
    //cout << "cluster: " << r_points[0].x << " " << r_points[0].y << endl;
    for (int i = 1; i < n; i++) {
        //cout << r_points[i].x << " " << r_points[i].y << endl;
        Point p = r_points[i];
        bool _new = true;
        int s = cluster_points.size();
        for (int j = 0; j < s; j++) {
            if (calculate_distance(p, cluster_points[j]) <= cluster_max_distance)
                _new = false;
        }
        if (_new) {
            cluster_points.push_back(p);
            //cout << "cluster: " << p.x << " " << p.y << endl;
            count++;
        }
    }

    cout << "pocet clusteru: " << count << endl;
}



int main() {


    start_timer();
    read_data();
    normalize_data();
    n = m;
    r_points = points;

    mean_shift();

    calculate_clusters();

    end_timer();
    print_elapsed_time();
    return 0;
} 