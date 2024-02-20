import gpt as g
import numpy as np

class gluon_measurement:
    def __init__(self, parameters):
        self.placeholder = parameters["placeholder"]

class get_gluonic_objects(gluon_measurement):
    def __init__(self, parameters):
        self.placeholder = parameters["placeholder"]

    #Fmunu is anti-symmetric, i.e. F01 = -F10. The diagonal parts are simply zeros
    def get_Fmunu(self, U):
        grid = U[0].grid
        Ex = g.qcd.gauge.field_strength(U, 0, 3) #FT03
        Ey = g.qcd.gauge.field_strength(U, 1, 3) #FT13
        Ez = g.qcd.gauge.field_strength(U, 2, 3) #FT23
        Bx = g.qcd.gauge.field_strength(U, 1, 2) #FT12
        By = g.qcd.gauge.field_strength(U, 2, 0) #FT20
        Bz = g.qcd.gauge.field_strength(U, 0, 1) #FT01
        #g.message("Fmunu done")
        return Ex,Ey,Ez,Bx,By,Bz

    def get_gluon_anomaly(self, U): #=1/4*F^a_{rho,sigma}*F^a_{rho,sigma}, Tr[t^i*t*j]=1/2*delta_{i,j}
        grid = U[0].grid
        Ex,Ey,Ez,Bx,By,Bz=self.get_Fmunu(U)
        gluon_anomaly = g.complex(grid)
        gluon_anomaly[:] = 0
        gluon_anomaly = g.eval(g.color_trace(Ex*Ex+Ey*Ey+Ez*Ez+Bx*Bx+By*By+Bz*Bz))
        #g.message("Trace Anomaly done.")
        return gluon_anomaly

    #The traceless part of gluonic Tmunu, which we name Umunu, is symmetric.
    def get_Umunu(self,U):#=F^a_{mu,rho}*F^a_{nu,rho}-delta_{mu,nu}*gluon_anomaly
        Umunu=[]
        Ex,Ey,Ez,Bx,By,Bz=self.get_Fmunu(U)
        gluon_anomaly =self.get_gluon_anomaly(U)
        U00 = g.eval(2.0*g.color_trace(g.eval(Bz*Bz+By*By+Ex*Ex)) - gluon_anomaly)
        Umunu.append(U00)
        U01 = g.eval(2.0*g.color_trace(g.eval(-By*Bx+Ex*Ey)))
        Umunu.append(U01)
        U02 = g.eval(2.0*g.color_trace(g.eval(-Bz*Bx+Ex*Ez)))
        Umunu.append(U02)
        U03 = g.eval(2.0*g.color_trace(g.eval(-Bz*Ey+By*Ez)))
        Umunu.append(U03)
        U11 = g.eval(2.0*g.color_trace(g.eval(Bz*Bz+Bx*Bx+Ey*Ey)) - gluon_anomaly)
        Umunu.append(U11)
        U12 = g.eval(2.0*g.color_trace(g.eval(-Bz*By+Ey*Ez)))
        Umunu.append(U12)
        U13 = g.eval(2.0*g.color_trace(g.eval(Bz*Ex-Bx*Ez)))
        Umunu.append(U13)
        U22 = g.eval(2.0*g.color_trace(g.eval(By*By+Bx*Bx+Ez*Ez)) - gluon_anomaly)
        Umunu.append(U22)
        U23 = g.eval(2.0*g.color_trace(g.eval(-By*Ex+Bx*Ey)))
        Umunu.append(U23)
        U33 = g.eval(2.0*g.color_trace(g.eval(Ex*Ex+Ey*Ey+Ez*Ez)) - gluon_anomaly)
        Umunu.append(U33)
        return Umunu

    def get_Utt(self,U):#Tr[\vec{E}^2-\vec{B}^2]
        Ex,Ey,Ez,Bx,By,Bz = self.get_Fmunu(U)
        gluon_anomaly = self.get_gluon_anomaly(U)
        U33 = g.eval(2.0*g.color_trace(g.eval(Ex*Ex+Ey*Ey+Ez*Ez)) - gluon_anomaly)
        return U33


    def get_Sg_z(self,U):
        Ex = g.qcd.gauge.field_strength(U, 0, 3)
        Ey = g.qcd.gauge.field_strength(U, 1, 3)
        Ax = g.eval(U[0] - g.adj(U[0]) + g.cshift(U[0],0,-1) + g.adj(g.cshift(U[0],0,-1)))
        Ax -= g.identity(Ax) * g.trace(Ax) / 3
        Ax @= 0.25 * (-1j) * Ax
        Ay = g.eval(U[1] - g.adj(U[1]) + g.cshift(U[1],1,-1) + g.adj(g.cshift(U[1],1,-1)))
        Ay -= g.identity(Ay) * g.trace(Ay) / 3
        Ay @= 0.25 * (-1j) * Ay
        Sg_z = g.trace(Ex * Ay - Ey * Ax)*2
        return g.eval(Sg_z)

    def get_EdotB(self,U):
        Ex,Ey,Ez,Bx,By,Bz=self.get_Fmunu(U)
        return g.eval(Ex*Bx + Ey*By + Ez*Bz)

    #gluon PDF using \mathrmcal{O}_0 of https://arxiv.org/pdf/1808.02077.pdf
    def get_gPDF(self,U):
        Ex,Ey,Ez,Bx,By,Bz = self.get_Fmunu(U)
        OEx = Ex
        OEy = Ey
        OBz = Bz
        for dz in range(0, self.zmax):
            OEx @= OEx * g.cshift(U[2], 2, 1)
            OEy @= OEy * g.cshift(U[2], 2, 1)
            OBz @= OBz * g.cshift(U[2], 2, 1)
        OEx @= OEx * g.cshift(Ex, 2, self.zmax)
        OEy @= OEy * g.cshift(Ey, 2, self.zmax)
        OBz @= OBz * g.cshift(Bz, 2, self.zmax)
        for dz in range(0, self.zmax):
            OEx @= OEx * g.cshift(U[2], 2, -1)
            OEy @= OEy * g.cshift(U[2], 2, -1)
            OBz @= OBz * g.cshift(U[2], 2, -1)
        O = 2.0 * g.color_trace(OEx+OEy-0.5*OBz)
        return O

    # mu -->                  *-->
    # nu        *^ ->- >      |  |
    # |          |     |  +   ^  v
    # v          |<-<- v      |  |
    #                          <--
    def get_Rect_P(self, U, mu, nu):
        P1 = g.eval(U[mu] * g.cshift(U[mu], mu, 1) * g.cshift(U[nu], mu, 2) * g.adj(g.cshift(g.cshift(U[mu], nu, 1), mu, 1)) * g.adj(g.cshift(U[mu], nu , 1))* g.adj(U[nu]))
        P2 = g.eval(U[mu] * g.cshift(U[nu],mu,1) * g.cshift(g.cshift(U[nu],mu,1), nu, 1) * g.adj(g.cshift(U[mu], nu, 2)) * g.adj(g.cshift(U[nu], nu, 1)) * g.adj(U[nu]))
        return g.eval(P1+P2)

    # mu -->                   -->
    # nu         ^ ->- >      |  |
    # |          |     |  +   ^  v
    # v         *|<-<- v      |  |
    #                         *<--
    def get_Rect_Q(self, U, mu, nu):
        Q1 = g.eval(g.adj(g.cshift(U[nu], nu, -1)) * g.cshift(U[mu], nu, -1) * g.cshift(g.cshift(U[mu], nu, -1), mu, 1) * g.cshift(g.cshift(U[nu], nu, -1), mu, 2) * g.adj(g.cshift(U[mu], mu, 1)) * g.adj(U[mu]))
        Q2 = g.eval(g.adj(g.cshift(U[nu], nu, -1)) * g.adj(g.cshift(U[nu], nu, -2)) * g.cshift(U[mu], nu, -2) * g.cshift(g.cshift(U[nu], mu, 1), nu, -2) * g.cshift(g.cshift(U[nu], mu, 1), nu, -1) * g.adj(U[mu]))
        return g.eval(Q1+Q2)

    # mu -->                  -->*
    # nu         ^ -> ->*     |  |
    # |          |     |  +   ^  v
    # v          |<-<- v      |  |
    #                          <--
    def get_Rect_S(self, U, mu, nu):
        S1 = g.eval(U[nu] * g.adj(g.cshift(g.cshift(U[mu], mu, -1), nu, 1)) * g.adj(g.cshift(g.cshift(U[mu], mu, -2), nu, 1)) * g.adj(g.cshift(U[nu], mu, -2)) * g.cshift(U[mu], mu, -2) * g.cshift(U[mu], mu, -1))
        S2 = g.eval(U[nu] * g.cshift(U[nu], nu, 1) * g.adj(g.cshift(g.cshift(U[mu], mu, -1), nu, 2)) * g.adj(g.cshift(g.cshift(U[nu], mu, -1), nu, 1)) * g.adj(g.cshift(U[nu], mu, -1)) * g.cshift(U[mu], mu, -1))
        return g.eval(S1+S2)

    # mu -->                    -->
    # nu         ^ -> ->       |  |
    # |          |     |  +    ^  v
    # v          |<-<- v*      |  |
    #                          <--*
    def get_Rect_R(self, U, mu, nu):
        R1 = g.eval(g.adj(g.cshift(U[mu], mu, -1)) * g.adj(g.cshift(U[mu], mu, -2)) * g.adj(g.cshift(g.cshift(U[nu], mu, -2), nu, -1)) * g.cshift(g.cshift(U[mu], mu, -2), nu, -1) * g.cshift(g.cshift(U[mu], mu, -1), nu, -1) * g.cshift(U[nu], nu, -1))
        R2 = g.eval(g.adj(g.cshift(U[mu], mu, -1)) * g.adj(g.cshift(g.cshift(U[nu], mu, -1), nu, -1)) * g.adj(g.cshift(g.cshift(U[nu], mu, -1), nu, -2)) * g.cshift(g.cshift(U[mu], mu, -1), nu, -2) * g.cshift(U[nu], nu, -2) * g.cshift(U[nu], nu, -1))
        return g.eval(R1+R2)

    def get_rectClover(self, U, mu, nu):
        rect_p = self.get_Rect_P(U, mu, nu)
        rect_q = self.get_Rect_Q(U, mu, nu)
        rect_r = self.get_Rect_R(U, mu, nu)
        rect_s = self.get_Rect_S(U, mu, nu)
        return g.eval(rect_p+rect_q+rect_r+rect_s)

    def get_Fmunu_imp(self, U, mu, nu):
        assert mu != nu
        Fmunu_plaq = g.qcd.gauge.field_strength(U, mu, nu)
        QmunuRect = self.get_rectClover(U, mu, nu)
        Fmunu_rect = g.eval(-1j * 0.0625 * (QmunuRect - g.adj(QmunuRect)))
        Fmunu_rect = g.eval(Fmunu_rect - 1./3. * g.trace(Fmunu_rect) * g.identity(Fmunu_rect))
        Fmunu = g.eval(5./3. * Fmunu_plaq - 1./3. * Fmunu_rect)
        #Fmunu = Fmunu - 1./3. * g.trace(Fmunu) * g.identity(Fmunu)
        return g.eval(Fmunu)

    def get_q(self, U): # q = g^2/(32*pi^2)epsilon(mu,nu,rho,sigma)Tr[Fmunu*Frhosigma]
        #compute the topological charge density
        Fmunu = g.qcd.gauge.field_strength(U, 0, 3)
        Frhodelta = g.qcd.gauge.field_strength(U, 1, 2)
        tmp = g.trace(g.eval(Fmunu * Frhodelta))

        Fmunu = g.qcd.gauge.field_strength(U, 1, 3)
        Frhodelta = g.qcd.gauge.field_strength(U, 2, 0)
        tmp += g.trace(g.eval(Fmunu * Frhodelta))

        Fmunu = g.qcd.gauge.field_strength(U, 2, 3)
        Frhodelta = g.qcd.gauge.field_strength(U, 0, 1)
        tmp += g.trace(g.eval(Fmunu * Frhodelta))
        #multiply with 1/4*pi^2
        tmp = g.eval(tmp)
        tmp /= (4. * np.pi * np.pi)
        return tmp

    def get_q_imp(self, U):
        #compute the improved topological charge density
        Fmunu = self.get_Fmunu_imp(U, 0, 3)
        Frhodelta = self.get_Fmunu_imp(U, 1, 2)
        tmp = g.trace(g.eval(Fmunu * Frhodelta))

        Fmunu = self.get_Fmunu_imp(U, 1, 3)
        Frhodelta = self.get_Fmunu_imp(U, 2, 0)
        tmp += g.trace(g.eval(Fmunu * Frhodelta))

        Fmunu = self.get_Fmunu_imp(U, 2, 3)
        Frhodelta = self.get_Fmunu_imp(U, 0, 1)
        tmp += g.trace(g.eval(Fmunu * Frhodelta))
        tmp = g.eval(tmp)
        tmp /= (4. * np.pi * np.pi)
        return tmp

