#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
    MatrixXd Ft = F_.transpose();

    x_ = F_ * x_;
    P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
    VectorXd y = z - (H_ * x_);
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd K = (P_ * Ht) * S.inverse();
    long x_size = x_.size();
    
    x_ = x_ + (K * y);
    P_ = (MatrixXd::Identity(x_size, x_size) - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    float px = x_(0);
    float py = x_(1);
    float vx = x_(2);
    float vy = x_(3);
    MatrixXd hx(3, 1);
    MatrixXd Ht = Hj_.transpose();
    MatrixXd S = Hj_ * P_ * Ht + Rextended_;
    MatrixXd K = (P_ * Ht) * S.inverse();
    static const float pi = 3.1415;
    long x_size = x_.size();

    hx << sqrt(px * px + py * py),
          atan2(py, px), 
          (px * vx + py * vy) / sqrt(px * px + py * py);

    VectorXd y = z - hx;

    while (y(1) < -pi) {
        y(1) += 2 * pi;
    }

    while (y(1) > pi) {
        y(1) -= 2 * pi;
    }

    x_ = x_ + (K * y);
    P_ = (MatrixXd::Identity(x_size, x_size) - K * Hj_) * P_;
}
