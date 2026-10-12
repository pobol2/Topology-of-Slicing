// Reporting adapter only. All clipping and all Newton steps are Geogram code.
#include <geogram/basic/common.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/mesh/mesh.h>
#include <geogram/voronoi/RVD.h>
#include <geogram/voronoi/RVD_callback.h>
#include <geogram/voronoi/generic_RVD_cell.h>
#include <geogram/basic/geometry.h>
#include <geogram/basic/geometry_nd.h>
#include <geogram/basic/numeric.h>
#include <geogram/mesh/mesh_reorder.h>
#include <exploragram/optimal_transport/optimal_transport_3d.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <vector>

#ifndef SEMIDISCRETE_WEIGHT_REGULARIZATION
#define SEMIDISCRETE_WEIGHT_REGULARIZATION 0.0
#endif
// Upstream HLBFGS stop factor (epsilon = factor/sqrt(n)) before handing off to Newton.
#ifndef SEMIDISCRETE_LBFGS_HANDOFF
#define SEMIDISCRETE_LBFGS_HANDOFF 0.05
#endif

// The cell-facet traversal below is adapted from Geogram (BSD-3-Clause).
// Copyright (c) 2000-2022 Inria. All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Inria nor the names of its contributors may be used
//   to endorse or promote products derived from this software without specific
//   prior written permission.
// * Neither the name of the ALICE Project-Team nor the names of its
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

using namespace GEO;
using Point = std::array<double,3>;
template<class T> void read(std::istream& s, T* p, size_t n) {
    s.read(reinterpret_cast<char*>(p), std::streamsize(n*sizeof(T)));
    if(!s) throw std::runtime_error("Truncated adapter input");
}
template<class T> void write(std::ostream& s, const T* p, size_t n) {
    s.write(reinterpret_cast<const char*>(p), std::streamsize(n*sizeof(T)));
    if(!s) throw std::runtime_error("Failed adapter output");
}
class ReportedOTM final : public OptimalTransportMap3d {
public:
    explicit ReportedOTM(Mesh* m,bool brio=false):OptimalTransportMap3d(m,"default",brio) {}
    index_t iterations() const { return current_iter_; }
    index_t empty_cells() const { return nbZ_; }
    double gradient_norm() const { return g_norm_; }
    std::string stats() const { return last_stats_; }
    static void report_layout() {
        // Read-only diagnostic metadata for observing an already running solver.
        // These are compiler ABI offsets, not an alternative convergence test.
        std::cout<<"{\"size\":"<<sizeof(ReportedOTM)
            <<",\"index_bytes\":"<<sizeof(index_t)
            <<",\"iterations\":"<<offsetof(ReportedOTM,current_iter_)
            <<",\"calls\":"<<offsetof(ReportedOTM,current_call_iter_)
            <<",\"gradient\":"<<offsetof(ReportedOTM,g_norm_)
            <<",\"empty\":"<<offsetof(ReportedOTM,nbZ_)
            <<",\"weights\":"<<offsetof(ReportedOTM,weights_)
            <<",\"stats\":"<<offsetof(ReportedOTM,last_stats_)<<"}"<<std::endl;
    }
};

template<class T> void restore_order(std::vector<T>& values,const GEO::vector<index_t>& order,size_t components=1) {
    auto copy=values;
    for(index_t i=0;i<order.size();++i) for(size_t d=0;d<components;++d) values[components*order[i]+d]=copy[components*i+d];
}

// Independent volume/mass measurements of the upstream clipped polyhedra.
// This callback never cuts geometry or changes solver state.
class CellReport final : public RVDPolyhedronCallback {
public:
    mutable std::vector<std::vector<Point>> vertices;
    mutable std::vector<double> volumes, masses, moments;
    mutable std::vector<uint32_t> piece_sites;
    mutable std::vector<uint64_t> piece_offsets{0};
    mutable std::vector<Point> piece_vertices;
    double slice_z;
    explicit CellReport(size_t n,double z):vertices(n),volumes(n,0),masses(n,0),moments(3*n,0),slice_z(z) {}
    void operator()(index_t site,index_t tetrahedron,const GEOGen::ConvexCell& c) const override {
        (void)tetrahedron;
        if(site>=vertices.size()) return; // Same air/infinite-site guard as Geogram's own callback.
        const GEOGen::Vertex* v0=nullptr;
        std::vector<Point> piece;
        for(index_t t=0;t<c.max_t();++t) if(c.triangle_is_used(t)) {
            const auto& v=c.triangle_dual(t);
            const double* p=v.point();
            vertices[site].push_back({p[0],p[1],p[2]});
            if(std::isfinite(slice_z)) piece.push_back({p[0],p[1],p[2]});
            if(!v0) v0=&v;
        }
        if(!v0) return;
        if(!piece.empty()) {
            double lo=piece[0][2],hi=lo;
            for(const auto& p:piece) {lo=std::min(lo,p[2]);hi=std::max(hi,p[2]);}
            if(lo<=slice_z && hi>=slice_z) {
                piece_sites.push_back(site);
                piece_vertices.insert(piece_vertices.end(),piece.begin(),piece.end());
                piece_offsets.push_back(piece_vertices.size());
            }
        }
        for(index_t face=0;face<c.max_v();++face) {
            signed_index_t t=c.vertex_triangle(face);
            if(t==-1) continue;
            GEOGen::ConvexCell::Corner first(index_t(t),c.find_triangle_vertex(index_t(t),face));
            auto corner=first;
            const auto* v1=&c.triangle_dual(first.t);
            const GEOGen::Vertex* v2=nullptr;
            const GEOGen::Vertex* v3=nullptr;
            do {
                v2=v3; v3=&c.triangle_dual(corner.t);
                if(v2 && v3!=v1 && v1!=v0 && v2!=v0 && v3!=v0) {
                    const double volume=Geom::tetra_volume<3>(v0->point(),v1->point(),v2->point(),v3->point());
                    volumes[site]+=volume;
                    masses[site]+=volume*(v0->weight()+v1->weight()+v2->weight()+v3->weight())/4.0;
                    const GEOGen::Vertex* vs[4]={v0,v1,v2,v3};
                    const double sum_rho=v0->weight()+v1->weight()+v2->weight()+v3->weight();
                    for(size_t d=0;d<3;++d) {
                        double sum_x=0, sum_rho_x=0;
                        for(const auto* v:vs) { sum_x+=v->point()[d];sum_rho_x+=v->weight()*v->point()[d]; }
                        // Exact integral of x*rho for affine rho on a tetrahedron.
                        moments[3*site+d]+=volume*(sum_rho*sum_x+sum_rho_x)/20.0;
                    }
                }
                c.move_to_next_around_vertex(corner);
            } while(corner!=first);
        }
    }
};

int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--layout") {
            ReportedOTM::report_layout(); return 0;
        }
        if(argc!=3) throw std::runtime_error("Usage: semidiscrete_adapter INPUT OUTPUT");
        GEO::initialize();
        CmdLine::import_arg_group("standard");
        CmdLine::import_arg_group("algo");
        std::ifstream input(argv[1],std::ios::binary);
        uint64_t h[5]; read(input,h,5);
        const index_t nv=index_t(h[0]),nt=index_t(h[1]),n=index_t(h[2]);
        const index_t maxiter=index_t(h[3]);
        const bool keep_vertices=h[4]!=0;
        double epsilon,slice_z; read(input,&epsilon,1);read(input,&slice_z,1);
        std::vector<double> positions(3*nv),sites(3*n),density(nv),initial_weights(n);
        std::vector<uint32_t> tetrahedra(4*nt);
        read(input,positions.data(),positions.size());read(input,tetrahedra.data(),tetrahedra.size());
        read(input,density.data(),density.size());read(input,sites.data(),sites.size());
        read(input,initial_weights.data(),initial_weights.size());
        GEO::vector<index_t> order,levels;
#ifdef SEMIDISCRETE_GEOGRAM_MULTILEVEL
        Numeric::random_reset(0);
        compute_BRIO_order(n,sites.data(),order,3,3,64,0.125,&levels);
        const auto original_sites=sites,original_weights=initial_weights;
        for(index_t i=0;i<n;++i) {
            initial_weights[i]=original_weights[order[i]];
            for(size_t d=0;d<3;++d) sites[3*i+d]=original_sites[3*order[i]+d];
        }
#endif
        Mesh domain(4);
        domain.vertices.create_vertices(nv);
        for(index_t v=0;v<nv;++v) for(index_t d=0;d<3;++d) domain.vertices.point_ptr(v)[d]=positions[3*v+d];
        for(index_t t=0;t<nt;++t) domain.cells.create_tet(tetrahedra[4*t],tetrahedra[4*t+1],tetrahedra[4*t+2],tetrahedra[4*t+3]);
        domain.cells.connect();
        Attribute<double> rho(domain.vertices.attributes(),"weight");
        for(index_t v=0;v<nv;++v) rho[v]=density[v];
        ReportedOTM ot(&domain,!order.empty());
        ot.set_points(n,sites.data(),3);
        for(index_t i=0;i<n;++i) ot.set_initial_weight(i,initial_weights[i]);
        ot.set_Newton(true);
#ifdef SEMIDISCRETE_DIRECT_SUPERLU
        ot.set_linear_solver(OT_SUPERLU);
#endif
        ot.set_regularization(SEMIDISCRETE_WEIGHT_REGULARIZATION);
        ot.set_epsilon(epsilon);
        ot.set_linsolve_epsilon(0.001); // Geogram default: avoid over-solving the gauge-null Hessian.
        ot.set_linsolve_maxiter(10000);
        ot.set_linesearch_maxiter(80);
        ot.set_linesearch_init_iter(0);
        ot.set_verbose(false);
        index_t initialization_iterations=0;
#ifdef SEMIDISCRETE_GEOGRAM_WARMSTART
        // Both stages are upstream optimisers; the adapter performs no updates.
        const auto warm_start=std::chrono::steady_clock::now();
        ot.set_Newton(false);
        ot.set_epsilon(SEMIDISCRETE_LBFGS_HANDOFF/std::sqrt(double(n)));
        std::cerr<<"ADAPTER_STAGE geogram_LBFGS_initialization"<<std::endl;
        ot.optimize(1000);
        initialization_iterations=ot.iterations();
        const std::string initialization_stats=ot.stats();
        std::vector<double> initialization_weights(n),initialization_masses(n);
        for(index_t i=0;i<n;++i) initialization_weights[i]=ot.weight(i);
        ot.compute_P1_Laplacian(initialization_weights.data(),nullptr,initialization_masses.data());
        double total=0,max_error=0;index_t empty_initial=0;
        for(double mass:initialization_masses) { total+=mass;if(!(mass>0)) ++empty_initial; }
        for(double mass:initialization_masses) max_error=std::max(max_error,std::abs(mass*double(n)/total-1));
        std::cerr<<std::setprecision(17)<<"GEOGRAM_WARMSTART {\"method\":\"Existing Geogram HLBFGS\",\"handoff_factor\":"
            <<SEMIDISCRETE_LBFGS_HANDOFF<<",\"iterations\":"
            <<initialization_iterations<<",\"runtime_seconds\":"
            <<std::chrono::duration<double>(std::chrono::steady_clock::now()-warm_start).count()
            <<",\"max_relative_mass_error\":"<<max_error<<",\"empty_cells\":"<<empty_initial
            <<",\"upstream_report\":"<<std::quoted(initialization_stats)<<"}"<<std::endl;
        if(empty_initial) throw std::runtime_error("Geogram L-BFGS initialization left empty cells; Newton not run");
        ot.set_Newton(true);
        ot.set_epsilon(epsilon);
#endif
        // Each Newton solve and every weight update are upstream Geogram code.
        std::cerr<<"ADAPTER_STAGE optimize"<<std::endl;
#ifdef SEMIDISCRETE_GEOGRAM_MULTILEVEL
        std::cerr<<"GEOGRAM_MULTILEVEL {\"method\":\"Existing Geogram multilevel Newton\",\"BRIO_seed\":0,\"BRIO_threshold\":64,\"BRIO_ratio\":0.125,\"weight_fit_degree\":2,\"weight_fit_neighbours\":20,\"levels\":[";
        for(index_t i=0;i<levels.size();++i) std::cerr<<(i?",":"")<<levels[i];
        std::cerr<<"]}"<<std::endl;
        ot.optimize_levels(levels,maxiter);
#else
        ot.optimize(maxiter);
#endif
        std::cerr<<"ADAPTER_STAGE optimized"<<std::endl;
        const auto iterations=ot.iterations()-initialization_iterations;
        const auto empty=ot.empty_cells();
        const auto gradient=ot.gradient_norm();
        const auto stats=ot.stats();
        std::vector<double> weights(n),own_masses(n),centroids(3*n),diameters(n);
        for(index_t i=0;i<n;++i) weights[i]=ot.weight(i);
        ot.compute_P1_Laplacian(weights.data(),nullptr,own_masses.data());
        std::cerr<<"ADAPTER_STAGE masses"<<std::endl;
        ot.compute_Laguerre_centroids(centroids.data());
        std::cerr<<"ADAPTER_STAGE geogram_centroids"<<std::endl;
        CellReport cells(n,slice_z);
        ot.RVD()->for_each_polyhedron(cells,false,false,false);
        std::cerr<<"ADAPTER_STAGE measured_cells"<<std::endl;
        if(!order.empty()) {
            restore_order(weights,order);restore_order(own_masses,order);restore_order(centroids,order,3);
            restore_order(cells.volumes,order);restore_order(cells.masses,order);restore_order(cells.moments,order,3);
            std::vector<std::vector<Point>> original_vertices(n);
            for(index_t i=0;i<n;++i) original_vertices[order[i]]=std::move(cells.vertices[i]);
            cells.vertices.swap(original_vertices);
            for(auto& site:cells.piece_sites) site=order[site];
        }
        std::vector<uint64_t> offsets(n+1,0);
        for(index_t i=0;i<n;++i) {
            auto& pts=cells.vertices[i];
            std::sort(pts.begin(),pts.end());
            pts.erase(std::unique(pts.begin(),pts.end()),pts.end());
            double max_d2=0;
            for(size_t a=0;a<pts.size();++a) for(size_t b=a+1;b<pts.size();++b) {
                double d2=0;
                for(size_t d=0;d<3;++d) d2+=(pts[a][d]-pts[b][d])*(pts[a][d]-pts[b][d]);
                max_d2=std::max(max_d2,d2);
            }
            diameters[i]=std::sqrt(max_d2);
            offsets[i+1]=offsets[i]+(keep_vertices?pts.size():0);
        }
        std::ofstream output(argv[2],std::ios::binary);
        uint64_t out_h[7]={n,iterations,empty,uint64_t(stats.size()),offsets.back(),cells.piece_sites.size(),cells.piece_vertices.size()};
        write(output,out_h,7);write(output,&gradient,1);
        write(output,stats.data(),stats.size());
        write(output,weights.data(),n);write(output,own_masses.data(),n);
        write(output,cells.volumes.data(),n);write(output,cells.masses.data(),n);
        write(output,centroids.data(),3*n);write(output,diameters.data(),n);
        for(index_t i=0;i<n;++i) for(size_t d=0;d<3;++d) cells.moments[3*i+d]/=cells.masses[i];
        write(output,cells.moments.data(),3*n);
        write(output,offsets.data(),n+1);
        if(keep_vertices) for(const auto& pts:cells.vertices) for(const auto& p:pts) write(output,p.data(),3);
        write(output,cells.piece_sites.data(),cells.piece_sites.size());
        write(output,cells.piece_offsets.data(),cells.piece_offsets.size());
        for(const auto& p:cells.piece_vertices) write(output,p.data(),3);
        std::cout<<"GEOGRAM_REPORT "<<stats<<std::endl;
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<e.what()<<std::endl;
        return 1;
    }
}
