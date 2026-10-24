// Native ns-3.44 propagation-kernel adapter, without Python bindings.
#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/mobility-module.h>
#include <ns3/propagation-module.h>
#include <cmath>
using namespace ns3;
struct Bridge {
 Ptr<ConstantPositionMobilityModel> bs=CreateObject<ConstantPositionMobilityModel>();
 Ptr<ConstantPositionMobilityModel> ue=CreateObject<ConstantPositionMobilityModel>();
 Ptr<ThreeGppUmaPropagationLossModel> los=CreateObject<ThreeGppUmaPropagationLossModel>();
 Ptr<ThreeGppUmaPropagationLossModel> nlos=CreateObject<ThreeGppUmaPropagationLossModel>();
 Ptr<LogDistancePropagationLossModel> simple=CreateObject<LogDistancePropagationLossModel>();
 Ptr<Node> bnode=CreateObject<Node>(),unode=CreateObject<Node>();
 Bridge(){
  bnode->AggregateObject(bs);unode->AggregateObject(ue);
  los->SetFrequency(3.5e9);nlos->SetFrequency(3.5e9);
  los->SetAttribute("ShadowingEnabled",BooleanValue(false));
  nlos->SetAttribute("ShadowingEnabled",BooleanValue(false));
  los->SetAttribute("EnforceParameterRanges",BooleanValue(false));
  nlos->SetAttribute("EnforceParameterRanges",BooleanValue(false));
  los->SetChannelConditionModel(CreateObject<AlwaysLosChannelConditionModel>());
  nlos->SetChannelConditionModel(CreateObject<NeverLosChannelConditionModel>());
  simple->SetPathLossExponent(2.2);simple->SetReference(1.,28.+20.*std::log10(3.5));
 }
};
extern "C" {
 void* bridge_create(){return new Bridge();}
 void bridge_delete(void* ptr){delete static_cast<Bridge*>(ptr);}
 void bridge_losses(void* ptr,const double* sites,int n,const double* users,int m,int mode,double* out){
  auto* b=static_cast<Bridge*>(ptr);
  for(int i=0;i<n;++i){
   b->bs->SetPosition(Vector(sites[2*i],sites[2*i+1],25.));
   for(int u=0;u<m;++u){
    b->ue->SetPosition(Vector(users[2*u],users[2*u+1],1.5));
    if(mode<2){
     double dx=users[2*u]-sites[2*i],dy=users[2*u+1]-sites[2*i+1];
     double horizontal=std::hypot(dx,dy);
     if(horizontal<10.){
      double ux=horizontal>0.?dx/horizontal:1.,uy=horizontal>0.?dy/horizontal:0.;
      b->ue->SetPosition(Vector(sites[2*i]+10.*ux,sites[2*i+1]+10.*uy,1.5));
     }
    }
    if(mode==0)out[i*m+u]=-b->los->CalcRxPower(0.,b->bs,b->ue);
    else if(mode==1)out[i*m+u]=-b->nlos->CalcRxPower(0.,b->bs,b->ue);
    else {
     double d2=std::hypot(sites[2*i]-users[2*u],sites[2*i+1]-users[2*u+1])+1.;
     double d3=std::sqrt(d2*d2+23.5*23.5);
     b->bs->SetPosition(Vector(0.,0.,0.));b->ue->SetPosition(Vector(d3,0.,0.));
     out[i*m+u]=-b->simple->CalcRxPower(0.,b->bs,b->ue);
     b->bs->SetPosition(Vector(sites[2*i],sites[2*i+1],25.));
    }
   }
  }
 }
}
