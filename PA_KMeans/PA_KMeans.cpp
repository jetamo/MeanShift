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
#include <sstream>
#include <algorithm>


using namespace std;

struct Point
{
    vector<float> coords;
};

float min_x = 999999999;
float max_x = -1;
float min_y = 999999999;
float max_y = -1;

float min_shift = 0.00001;

float radius = 0.1f;
int n = 200;
int m = 0;

int dimensions = 0;

//string fileName = "mnist_test.csv";
string fileName = "twodee.csv";

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
    ifstream file(fileName);
    string line;
    while (getline(file, line)) {
        Point point;
        stringstream lineStream(line);
        string cell;
        while (getline(lineStream, cell, ',')) {
            float temp_coord = stod(cell);
            point.coords.push_back(temp_coord);
        }
        points.push_back(point);
        m++;
    }
}

float lerp(float v, float a, float b) {
    if (a == b)
        return 0.f;
    else
        return (v - a) / (b - a);
}

void normalize_data() {
    vector<Point> normalized_points;
    dimensions = points[0].coords.size();
    vector<float> min_values(dimensions, numeric_limits<float>::max());
    vector<float> max_values(dimensions, numeric_limits<float>::lowest());
    for (Point point : points) {
        for (int i = 0; i < dimensions; i++) {
            min_values[i] = min(min_values[i], point.coords[i]);
            max_values[i] = max(max_values[i], point.coords[i]);
        }
    }
    for (Point point : points) {
        Point normalized_point;
        for (int i = 0; i < dimensions; i++) {
            float normalized_coord = lerp(point.coords[i], min_values[i], max_values[i]);
            normalized_point.coords.push_back(normalized_coord);
        }
        normalized_points.push_back(normalized_point);
        cout << normalized_point.coords[0] << " " << normalized_point.coords[1] << endl;
    }
    points = normalized_points;
}


float calculate_distance(Point a, Point b) {
    float _sum = 0;
    for (int i = 0; i < dimensions; i++) {
        float diff = a.coords[i] - b.coords[i];
        _sum += diff * diff;
    }
    return sqrt(_sum);
}


/*
Point calculate_shift(Point p1) {
    bool done = false;

    while (!done) {

        vector<float> sum(dimensions, 0.f);

        float sum2 = 0.f;

        Point p_shifted;
        p_shifted.coords.resize(dimensions, 0.f);
//#pragma omp parallel for
        for (int i = 0; i < m; i++) {
            Point p2 = points[i];
            float distance = calculate_distance(p1, p2);
            if (distance > radius) {
                continue;
            }

            for (int j = 0; j < dimensions; j++) {
                //sum1_x +=(1 / pow((sqrt(2 * M_PI)) * 1, 2))          * exp((-1) * (pow(distance, 2) / (2 * pow(1, 2)))) * p2.x;
                sum[j] +=  (1 / pow((sqrt(2 * M_PI)) * 1, dimensions)) * exp((-1) * (pow(distance, 2) / (2 * pow(1, 2)))) * p2.coords[j];
            }

            
            sum2 +=       (1 / pow((sqrt(2 * M_PI)) * 1, dimensions)) * exp((-1) * (pow(distance, 2) / (2 * pow(1, 2))));

        }

        for (int i = 0; i < dimensions; i++) {
            float shifted_coord = sum[i] / sum2;
            p_shifted.coords[i] = shifted_coord;
            p1.coords[i] = shifted_coord;

        }

        float tmp_distance = calculate_distance(p1, p_shifted);
        
        if (tmp_distance < min_shift)
            done = true;
    }
    return p1;
}
*/

Point calculate_shift(Point p) {
    bool done = false;

    while (!done) {
        vector<float> sum1(dimensions, 0.f);
        float sum2 = 0.f;
#pragma omp parallel for
        for (int i = 0; i < m; i++) {
            Point p2 = points[i];
            float distance = calculate_distance(p, p2);
            if (distance > radius) {
                continue;
            }

            float value = (1 / pow((sqrt(2 * M_PI)) * 1, 2)) * exp((-1) * (pow(distance, 2) / (2 * pow(1, 2))));
            sum2 += value;

            for (int j = 0; j < dimensions; j++) {
                sum1[j] += value * p2.coords[j];
            }
        }

        std::vector<float> shifted_coords(dimensions);
        for (int j = 0; j < dimensions; j++) {
            shifted_coords[j] = sum1[j] / sum2;
        }

        Point p_shifted;
        p_shifted.coords = shifted_coords;

        float tmp_distance = calculate_distance(p, p_shifted);

        if (tmp_distance < min_shift)
            done = true;

        p.coords = shifted_coords;
    }

    return p;
}
void mean_shift() {
#pragma omp parallel for
    for (int l = 0; l < n; l++) {
        Point p = calculate_shift(r_points[l]);
        r_points[l] = p;
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