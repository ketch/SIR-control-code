! =====================================================
subroutine rpn2(ixy,maxm,num_eqn,num_waves,num_aux,mbc,mx,ql,qr,auxl,auxr,wave,s,amdq,apdq)
! =====================================================
! Riemann-solver for the advection equation
!    q_t  +  u*q_x + v*q_y = 0
! where u and v are a given velocity field.

! waves: 1
! equations: 1
! aux fields: 2

! Conserved quantities:
!       1 q

! Auxiliary variables:
!         1  x_velocity
!         2  y_velocity

! solve Riemann problems along one slice of data.
! This data is along a slice in the x-direction if ixy=1
!                            or the y-direction if ixy=2.

! On input, ql contains the state vector at the left edge of each cell
!           qr contains the state vector at the right edge of each cell

! On output, wave contains the waves, s the speeds,
! and amdq, apdq the left-going and right-going flux differences,
! respectively.  Note that in this advective form, the sum of
! amdq and apdq is not equal to a difference of fluxes except in the
! case of constant velocities.

! Note that the i'th Riemann problem has left state qr(i-1,:)
!                                    and right state ql(i,:)
! From the basic clawpack routines, this routine is called with ql = qr

    implicit double precision (a-h,o-z)

    integer, intent(in) :: num_eqn, num_aux, num_waves, ixy, maxm, mbc, mx
    double precision, intent(in out) :: wave(num_eqn, num_waves, 1-mbc:maxm+mbc)
    double precision, intent(in out) :: s(num_waves, 1-mbc:maxm+mbc)
    double precision, intent(in) ::  ql(num_eqn, 1-mbc:maxm+mbc)
    double precision, intent(in) ::  qr(num_eqn, 1-mbc:maxm+mbc)
    double precision, intent(in out) :: apdq(num_eqn,1-mbc:maxm+mbc)
    double precision, intent(in out) :: amdq(num_eqn,1-mbc:maxm+mbc)
    double precision, intent(in) :: auxl(num_aux,1-mbc:maxm+mbc)
    double precision, intent(in) :: auxr(num_aux,1-mbc:maxm+mbc)

    integer :: i
    double precision :: x, y, sigma

    common /cparam/ beta, gam

    do i = 2-mbc, mx+mbc

        sigma = 0.5d0*(auxl(3,i)+auxr(3,i-1))
        if (ixy == 1) then
            x = 0.5d0*(auxl(1,i)+auxr(1,i-1))
            y = auxl(2,i)
            s(1,i) = sigma*gam*x*y
        else
            x = auxl(1,i)
            y = 0.5d0*(auxl(2,i)+auxr(2,i-1))
            s(1,i) = gam*y * (1.d0 - sigma*x)
        endif
        wave(1,1,i) = ql(1,i) - qr(1,i-1)

        amdq(1,i) = dmin1(s(1,i), 0.d0) * wave(1,1,i)
        apdq(1,i) = dmax1(s(1,i), 0.d0) * wave(1,1,i)
    end do

    return
end subroutine rpn2
