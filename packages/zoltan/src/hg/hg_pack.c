/*****************************************************************************
 * Zoltan Library for Parallel Applications                                  *
 * Copyright (c) 2000,2001,2002, Sandia National Laboratories.               *
 * For more info, see the README file in the top-level Zoltan directory.     *
 *****************************************************************************/
/*****************************************************************************
 * CVS File Information :
 *    $RCSfile$
 *    $Author$
 *    $Date$
 *    $Revision$
 ****************************************************************************/

#ifdef __cplusplus
/* if C++, define the rest of this header file as extern C */
extern "C" {
#endif

#include "hypergraph.h"

static ZOLTAN_HG_PACKING_FN packing_mxp; /* maximal packing */
static ZOLTAN_HG_PACKING_FN packing_rep; /* random edge packing */
static ZOLTAN_HG_PACKING_FN packing_rrp; /* random, random edge packing */
static ZOLTAN_HG_PACKING_FN packing_rhp; /* random, heavy edge packing */
static ZOLTAN_HG_PACKING_FN packing_grp; /* greedy packing */
static ZOLTAN_HG_PACKING_FN packing_lhp; /* locally heaviest packing */
static ZOLTAN_HG_PACKING_FN packing_pgp; /* path growing packing */
static ZOLTAN_HG_PACKING_FN packing_aug2;/* augmenting path; length 2 */
static ZOLTAN_HG_PACKING_FN packing_aug3;/* augmenting path; length 3 */

/****************************************************************************/

int Zoltan_HG_Set_Packing_Fn(HGPartParams *hgp)
{
  int found = 0;
  char *str;
  static int srand_set = 0;
  if (srand_set == 0)
  { srand_set = 1 ;
    srand ((unsigned long) RANDOM_SEED) ;
  }

  str = hgp->redm_str;
  if      (!strcasecmp(str, "mxp"))  hgp->packing = packing_mxp;
  else if (!strcasecmp(str, "rep"))  hgp->packing = packing_rep;
  else if (!strcasecmp(str, "rrp"))  hgp->packing = packing_rrp;
  else if (!strcasecmp(str, "rhp"))  hgp->packing = packing_rhp;
  else if (!strcasecmp(str, "grp"))  hgp->packing = packing_grp;
  else if (!strcasecmp(str, "lhp"))  hgp->packing = packing_lhp;
  else if (!strcasecmp(str, "pgp"))  hgp->packing = packing_pgp;
  else                               hgp->packing = NULL;

  if (hgp->packing) {

    found = 1;

    /* 
     * If reduction method is a packing, set the improvement and 
     * edge weight scaling functions accordingly.
     */
    str = hgp->rli_str;
    if      (!strcasecmp(str, "aug2"))  hgp->packing_rli = packing_aug2;
    else if (!strcasecmp(str, "aug3"))  hgp->packing_rli = packing_aug3;
    else                                hgp->packing_rli = NULL;

  }
  return found;
}

/****************************************************************************/

int Zoltan_HG_Packing (ZZ *zz, HGraph *hg, Packing pack, HGPartParams *hgp)
{
  int   limit=0 ;   /* reserved for future use */
  int   ierr = ZOLTAN_OK;
  float *old_ewgt=NULL, *new_ewgt;
  char  *yo = "Zoltan_HG_Packing";

  ZOLTAN_TRACE_ENTER(zz, yo);

  if (hg->vwgt && hgp->ews)
  { if (!(new_ewgt = (float *) ZOLTAN_MALLOC (sizeof (float) * hg->nEdge)))
    { ZOLTAN_PRINT_ERROR(zz->Proc, yo, "Insufficient memory.");
      ZOLTAN_TRACE_EXIT(zz, yo);
      return ZOLTAN_MEMERR;
    }
    Zoltan_HG_Scale_HGraph_Weight (zz, hg, new_ewgt);
    old_ewgt = hg->ewgt;
    hg->ewgt = new_ewgt;
  }

  ierr = hgp->packing(zz,hg,pack, limit);
  if (ierr != ZOLTAN_OK && ierr != ZOLTAN_WARN) {
    goto End;
  }

  /* Optimization */
  if (hgp->packing_rli != NULL)
   ierr = hgp->packing_rli (zz,hg,pack,limit);

End:
  if (hg->vwgt && hgp->ews)
  { hg->ewgt = old_ewgt;
    ZOLTAN_FREE ((void **) &new_ewgt);
  }
  ZOLTAN_TRACE_EXIT(zz, yo);
  return ierr;
}

/****************************************************************************/

static int packing_mxp (ZZ *zz, HGraph *hg, Packing pack, int limit)
{                    /* limit is defined for future use */
  int i, j ;

  for (i = 0 ; i < hg->nVtx ; i++)
    pack[i] = i ;

  for (i = 0 ; i < hg->nEdge ; i++)
  { for (j = hg->hindex[i] ; j < hg->hindex[i+1] ; j++)
      if (pack[hg->hvertex[j]] != hg->hvertex[j])
        break ;
    if (j == hg->hindex[i+1])    /* if true, all vertices free for packing */
    { for (j = hg->hindex[i] ; j < hg->hindex[i+1]-1 ; j++)
        pack[hg->hvertex[j]] = hg->hvertex[j+1] ;
      pack[hg->hvertex[j]] = hg->hvertex[hg->hindex[i]] ;
    }
  }
  return ZOLTAN_OK ;
}

/****************************************************************************/

static int packing_rep (ZZ *zz, HGraph *hg, Packing pack, int limit)
{                              /* limit is defined for future use */
  int i, j, *edges=NULL, edge, random ;
  char *yo = "packing_rep" ;

  if (!(edges = (int *) ZOLTAN_MALLOC (sizeof (int) * hg->nEdge)))
  { ZOLTAN_PRINT_ERROR(zz->Proc, yo, "Insufficient memory.");
    return ZOLTAN_MEMERR;
  }
  for (i = 0 ; i < hg->nEdge ; i++)
    edges[i] = i ;
  for (i = 0 ; i < hg->nVtx ;  i++)
    pack[i]  = i ;

  for (i = hg->nEdge ; i > 0 ; i--)
  { edge = edges[random=rand()%i] ;
    edges[random] = edges[i-1] ;

    for (j = hg->hindex[edge] ; j < hg->hindex[edge+1] ; j++)
      if (pack[hg->hvertex[j]] != hg->hvertex[j])
        break ;
    if (j == hg->hindex[edge+1])   /* if true, all vertices free for packing */
    { for (j = hg->hindex[edge] ; j < hg->hindex[edge+1]-1 ; j++)
        pack[hg->hvertex[j]] = hg->hvertex[j+1] ;
      pack[hg->hvertex[j]] = hg->hvertex[hg->hindex[edge]] ;
    }
  }
  ZOLTAN_FREE ((void **) &edges) ;
  return ZOLTAN_OK ;
}

/****************************************************************************/

static int packing_rrp (ZZ *zz, HGraph *hg, Packing pack, int limit)
{                                      /* limit is defined for future use */
  int i, j, k, edge, random, *vertices=NULL, vertex ;
  int *del_edges=NULL, count ;
  char *yo = "packing_rrp" ;

  if (!(vertices  = (int *) ZOLTAN_MALLOC (sizeof (int) * hg->nVtx))  ||
      !(del_edges = (int *) ZOLTAN_MALLOC (sizeof (int) * hg->nEdge))  )
  { ZOLTAN_FREE ((void **) &vertices) ;
    ZOLTAN_FREE ((void **) &del_edges) ;
    ZOLTAN_PRINT_ERROR(zz->Proc, yo, "Insufficient memory.");
    return ZOLTAN_MEMERR;
  }
  for (i = 0 ; i < hg->nVtx ;  i++)
    vertices[i] = pack[i] = i ;
  for (i = 0 ; i < hg->nEdge ; i++)
    del_edges[i] = 0 ;

  for (i = hg->nVtx ; i > 0 ; i--)
  { vertex = vertices[random=rand()%i] ;
    vertices[random] = vertices[i-1] ;
    if (pack[vertex] != vertex)
      continue ;          /* vertex already packed, move on */

    count = 0 ;            /* count will be number of viable edges */
    for (k = hg->vindex[vertex] ; k < hg->vindex[vertex+1] ; k++)
    { edge = hg->vedge[k] ;
      if (del_edges[edge] == 1)
        continue ;       /* edge has been deleted for use already */

      for (j = hg->hindex[edge] ; j < hg->hindex[edge+1] ; j++)
        if (pack[hg->hvertex[j]] != hg->hvertex[j])
          break ;
      if (j == hg->hindex[edge+1])
        count++ ;                 /* found a possible edge */
      else
        del_edges[edge] = 1 ;     /* edge unusable, delete it from consideration */
    }
    if (count == 0)
      continue ;                  /* vertex has no free edges available */

    random = rand() % count;      /* randomly select from available edges */
    for (k = hg->vindex[vertex] ; k < hg->vindex[vertex+1] ; k++)
    { edge = hg->vedge[k] ;
      if (del_edges[edge] == 0 && --count == random)
      { for (j = hg->hindex[edge] ; j < hg->hindex[edge+1]-1 ; j++)
          pack[hg->hvertex[j]] = hg->hvertex[j+1] ;
        pack[hg->hvertex[j]] = hg->hvertex[hg->hindex[edge]] ;
      }
      del_edges[edge] = 1 ;
    }
  }
  ZOLTAN_FREE ((void **) &vertices) ;
  ZOLTAN_FREE ((void **) &del_edges) ;
  return ZOLTAN_OK ;
}

/****************************************************************************/

static int packing_rhp (ZZ *zz, HGraph *hg, Packing pack, int limit)
{                                   /* limit is defined for future use */
   int   i, j, k, *vertices=NULL, *del_edges=NULL, vertex, edge,
         number, best_edge, best_size;
   float best_ewgt;
   char  *yo = "packing_hep" ;

   if (!(vertices  = (int *) ZOLTAN_MALLOC (sizeof (int) * hg->nVtx))  ||
       !(del_edges = (int *) ZOLTAN_MALLOC (sizeof (int) * hg->nEdge))  )
      {
      ZOLTAN_FREE ((void **) &vertices) ;
      ZOLTAN_FREE ((void **) &del_edges) ;
      ZOLTAN_PRINT_ERROR(zz->Proc, yo, "Insufficient memory.");
      return ZOLTAN_MEMERR;
      }
   for (i = 0 ; i < hg->nVtx ; i++)
      pack[i] = vertices[i] = i ;
   for (i=0; i<hg->nEdge; i++)
      del_edges[i] = 0;

   for (i = hg->nVtx ; i > 0 ; i--)
      {
      vertex = vertices[number=rand()%i] ;
      vertices[number] = vertices[i-1] ;
      if (pack[vertex] != vertex)
         continue ;            /* vertex is already matched, move on */

      best_edge = best_size = -1;
      best_ewgt = -1.0 ;
      for (j = hg->vindex[vertex] ; j < hg->vindex[vertex+1] ; j++)
      {
         int size ;
         edge = hg->vedge[j];
         size = hg->hindex[edge+1] - hg->hindex[edge] ;
         if (del_edges[edge]==0 && ((!(hg->ewgt) && size < best_size)
          || (hg->ewgt && (hg->ewgt[edge] >  best_ewgt
                       || (hg->ewgt[edge] == best_ewgt && size < best_size)))))
            {
            best_edge = edge ;
            best_ewgt = hg->ewgt[best_edge] ;
            best_size = hg->hindex[best_edge+1]-hg->hindex[best_edge];
            }
      }
      if (best_edge == -1)
         continue ;

      for (j = hg->hindex[best_edge] ; j < hg->hindex[best_edge+1] ; j++)
        for (k=hg->vindex[hg->hvertex[j]]; k<hg->vindex[hg->hvertex[j]+1]; k++)
          del_edges[hg->vedge[k]] = 1;

      for (j = hg->hindex[best_edge] ; j < hg->hindex[best_edge+1]-1 ; j++)
         pack[hg->hvertex[j]] = hg->hvertex[j+1] ;
      pack[hg->hvertex[j]] = hg->hvertex[hg->hindex[best_edge]] ;
      }
   ZOLTAN_FREE ((void **) &vertices) ;
   ZOLTAN_FREE ((void **) &del_edges) ;
   return ZOLTAN_OK ;
}

/****************************************************************************/

static int packing_grp (ZZ *zz, HGraph *hg, Packing pack, int limit)
{                                         /* limit is defined for future use */
  int   i, j, *size=NULL, *sorted=NULL;
  char *yo = "packing_grp" ;

  for (i=0; i<hg->nVtx; i++)
    pack[i] = i;

/* Sort the hyperedges according to their weight and size */
  if (!(size   = (int *) ZOLTAN_MALLOC (sizeof (int) * hg->nEdge)) ||
      !(sorted = (int *) ZOLTAN_MALLOC (sizeof (int) * hg->nEdge))  )
     {
     ZOLTAN_FREE ((void **) &size) ;
     ZOLTAN_FREE ((void **) &sorted) ;
     ZOLTAN_PRINT_ERROR(zz->Proc, yo, "Insufficient memory.");
     return ZOLTAN_MEMERR;
     }
  for (i=0; i<hg->nEdge; i++)
     size[i] = -(hg->hindex[i+1]-hg->hindex[i]);
  for (i=0; i<hg->nEdge; i++)
     sorted[i] = i;
  quicksort_dec_float_int(sorted,hg->ewgt,size,0,hg->nEdge-1);
  ZOLTAN_FREE ((void **) &size);

/* Match hyperedges along decreasing weight */
  for (i=0; i<hg->nEdge; i++)
  { for (j=hg->hindex[sorted[i]]; j<hg->hindex[sorted[i]+1]; j++)
      if (pack[hg->hvertex[j]] != hg->hvertex[j])
        break;
    if (j == hg->hindex[sorted[i]+1])
    { for (j=hg->hindex[sorted[i]]; j<hg->hindex[sorted[i]+1]-1; j++)
        pack[hg->hvertex[j]] = hg->hvertex[j+1];
      pack[hg->hvertex[j]] = hg->hvertex[hg->hindex[sorted[i]]];
  } }
  ZOLTAN_FREE ((void **) &sorted);
  return ZOLTAN_OK;
}

/****************************************************************************/

static int lhp_pack (ZZ *zz, HGraph *hg, int edge, int *del_edge, int *Vindex,
                     Packing pack)
{ int	i, j, vertex, *Vindex_old, next_edge, done=0;
  char *yo = "lhp_pack" ;

  if (!(Vindex_old = (int *) ZOLTAN_MALLOC
     (sizeof(int)*(hg->hindex[edge+1]-hg->hindex[edge]))))
     {
     ZOLTAN_PRINT_ERROR(zz->Proc, yo, "Insufficient memory.");
     return ZOLTAN_MEMERR;
     }
  for (i=0; i<hg->hindex[edge+1]-hg->hindex[edge]; i++)
    Vindex_old[i] = Vindex[hg->hvertex[i+hg->hindex[edge]]];

  while (hg->ewgt && del_edge[edge]==0 && done==0)
  { done = 1;
    for (i=hg->hindex[edge]; i<hg->hindex[edge+1]; i++)
    { vertex = hg->hvertex[i];
      if (Vindex[vertex] < hg->vindex[vertex+1])
      { next_edge = hg->vedge[Vindex[vertex]];
        (Vindex[vertex])++;
        if (hg->ewgt[next_edge] > hg->ewgt[edge])
          lhp_pack(zz,hg,next_edge,del_edge,Vindex,pack);
        done=0;
  } } }

  if (del_edge[edge] == 0)
  { for (i=hg->hindex[edge]; i<hg->hindex[edge+1]-1; i++)
      pack[hg->hvertex[i]] = hg->hvertex[i+1];
    pack[hg->hvertex[i]] = hg->hvertex[hg->hindex[edge]];
    for (i=hg->hindex[edge]; i<hg->hindex[edge+1]; i++)
    { vertex = hg->hvertex[i];
      for (j=hg->vindex[vertex]; j<hg->vindex[vertex+1]; j++)
        del_edge[hg->vedge[j]] = 1;
    }
  }
  else
    for (i=0; i<hg->hindex[edge+1]-hg->hindex[edge]; i++)
    { vertex = hg->hvertex[i+hg->hindex[edge]];
      if (pack[vertex] == vertex)
        Vindex[vertex] = Vindex_old[i];
    }

  ZOLTAN_FREE((void **) &Vindex_old);
  return ZOLTAN_OK;
}

static int packing_lhp (ZZ *zz, HGraph *hg, Packing pack, int limit)
{                                      /* limit is defined for future use */
  int	i, *Vindex=NULL, *del_edge=NULL;
  char *yo = "packing_lhp" ;

  for (i=0; i<hg->nVtx; i++)
    pack[i] = i;

  if (!(del_edge = (int *) ZOLTAN_MALLOC (sizeof (int) * hg->nEdge))  ||
      !(Vindex   = (int *) ZOLTAN_MALLOC (sizeof (int) * (hg->nVtx+1))) )
     {
     ZOLTAN_FREE ((void **) &del_edge) ;
     ZOLTAN_FREE ((void **) &Vindex) ;
     ZOLTAN_PRINT_ERROR(zz->Proc, yo, "Insufficient memory.");
     return ZOLTAN_MEMERR;
     }
  for (i=0; i<hg->nEdge; i++)
     del_edge[i] = 0 ;
  memcpy(Vindex,hg->vindex,(hg->nVtx+1)*sizeof(int));

  for (i=0; i<hg->nEdge; i++)
    if (del_edge[i] == 0)
      lhp_pack(zz,hg,i,del_edge,Vindex,pack);

  ZOLTAN_FREE ((void **) &del_edge);
  ZOLTAN_FREE ((void **) &Vindex);
  return ZOLTAN_OK;
}

/****************************************************************************/

static int packing_pgp (ZZ *zz, HGraph *hg, Packing pack, int limit)
{                         /* limit is defined for future use */
 int   i, j, k, vertex, edge, *pack1=pack, *pack2=NULL, *Pack=pack,
        *taken_edge=NULL, *taken_vertex=NULL, cur_edge, best_edge;
  float	best_weight, w1=0.0, w2=0.0;
  char *yo = "packing_pgp" ;

  if (!(pack2        = (int *) ZOLTAN_MALLOC (sizeof (int) * hg->nVtx))  ||
      !(taken_edge   = (int *) ZOLTAN_MALLOC (sizeof (int) * hg->nEdge)) ||
      !(taken_vertex = (int *) ZOLTAN_MALLOC (sizeof (int) * hg->nVtx))   )
     {
     ZOLTAN_FREE ((void **) &pack2) ;
     ZOLTAN_FREE ((void **) &taken_edge) ;
     ZOLTAN_FREE ((void **) &taken_vertex) ;
     ZOLTAN_PRINT_ERROR(zz->Proc, yo, "Insufficient memory.");
     return ZOLTAN_MEMERR;
     }
  for (i=0; i<hg->nEdge; i++)
     taken_edge[i]=0 ;
  for (i=0; i< hg->nVtx; i++)
     taken_vertex[i]=0 ;
  for (i=0; i<hg->nVtx; i++)
     pack1[i] = pack2[i] = i;

  for (i=0; i<hg->nEdge; i++)
    if (taken_edge[i] == 0)
    { cur_edge = i;
      taken_edge[cur_edge] = 1;
      while (cur_edge >= 0)
      { for (j=hg->hindex[cur_edge]; j<hg->hindex[cur_edge+1]-1; j++)
          Pack[hg->hvertex[j]] = hg->hvertex[j+1];
        Pack[hg->hvertex[j]] = hg->hvertex[hg->hindex[cur_edge]];
        if (Pack == pack1)
        { w1 += (hg->ewgt?hg->ewgt[cur_edge]:1.0);
          Pack = pack2;
        }
        else
        { w2 += (hg->ewgt?hg->ewgt[cur_edge]:1.0);
          Pack = pack1;
        }

        best_weight = 0.0;
        best_edge = -1;
        for (j=hg->hindex[cur_edge]; j<hg->hindex[cur_edge+1]; j++)
          if (taken_vertex[(vertex=hg->hvertex[j])] == 0)
          { for (k=hg->vindex[vertex]; k<hg->vindex[vertex+1]; k++)
              if (taken_edge[(edge=hg->vedge[k])]==0)
              { if ((hg->ewgt?hg->ewgt[edge]:1.0)>best_weight)
                { best_edge = edge;
                  best_weight = (hg->ewgt?hg->ewgt[best_edge]:1.0);
                }
                taken_edge[edge] = 1;
              }
            taken_vertex[vertex] = 1;
          }
        cur_edge = best_edge;
    } }

  if (w1 < w2)
    memcpy(pack,pack2,(hg->nVtx) * sizeof(int));

  ZOLTAN_FREE ((void **) &pack2);
  ZOLTAN_FREE ((void **) &taken_edge);
  ZOLTAN_FREE ((void **) &taken_vertex);
  return ZOLTAN_OK;
}

/****************************************************************************/
static int packing_aug2 (ZZ *zz, HGraph *hg, Packing pack, int limit)
{
/* Placeholder for packing_aug2 */
  return ZOLTAN_OK;
}

/****************************************************************************/
static int packing_aug3 (ZZ *zz, HGraph *hg, Packing pack, int limit)
{
/* Placeholder for packing_aug3 */
  return ZOLTAN_OK;
}

/****************************************************************************/
#ifdef __cplusplus
} /* closing bracket for extern "C" */
#endif
