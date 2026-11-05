/* Static adapter: demonstration records + browser-local activity. No server. */
(function () {
  'use strict';
  const KEY='fieldwise-actions-2026-11-01-v1';
  const h=window.FIELDWISE_DATA.hcps, drivers=window.FIELDWISE_DATA.drivers;
  const ids=new Map(h.map(row=>[row.hcp_id,row]));
  function read(){try{const parsed=JSON.parse(localStorage.getItem(KEY)||'[]');return Array.isArray(parsed)?parsed:[]}catch{return []}}
  function write(events){try{localStorage.setItem(KEY,JSON.stringify(events))}catch{throw Error('Browser storage is unavailable. Your action could not be saved.')}}
  function states(){const result={};read().forEach(e=>{if(['planned','contacted','deferred'].includes(e.event))result[e.hcp_id]=e});return result}
  window.PagesBackend={
    get:async function(url){
      const parsed=new URL(url,'https://demo.invalid'), q=parsed.searchParams;
      if(parsed.pathname==='/api/hcps'){
        const kind=q.get('list')||'uplift', status=states();
        let rows=h.filter(r=>r[kind+'_flagged']).map(r=>({...r,status:status[r.hcp_id]?.event||'new'}));
        if(q.get('territory'))rows=rows.filter(r=>r.territory===q.get('territory'));
        if(q.get('stage'))rows=rows.filter(r=>r.hcp_stage===q.get('stage'));
        if(q.get('q'))rows=rows.filter(r=>r.hcp_id.toLowerCase().includes(q.get('q').trim().toLowerCase()));
        if(q.get('tier1')==='1')rows=rows.filter(r=>r.tier1);
        if(q.get('new_only')==='1')rows=rows.filter(r=>!r.on_rep_plan);
        rows.sort((a,b)=>a[kind==='uplift'?'uplift_rank':'current_rank']-b[kind==='uplift'?'uplift_rank':'current_rank']);
        return {rows,total:rows.length};
      }
      if(parsed.pathname.startsWith('/api/hcps/')){
        const id=decodeURIComponent(parsed.pathname.split('/').pop());
        if(!ids.has(id))throw Error('HCP not found.');
        return {hcp:ids.get(id),drivers:drivers[id]||[],status:states()[id]||{}};
      }
      if(parsed.pathname==='/api/summary'){
        const current=new Set(h.filter(r=>r.current_flagged).map(r=>r.hcp_id));
        const uplift=new Set(h.filter(r=>r.uplift_flagged).map(r=>r.hcp_id));
        const plan=new Set(h.filter(r=>r.on_rep_plan).map(r=>r.hcp_id));
        const events=read();const summary={universe:h.length,current:current.size,uplift:uplift.size,
          overlap:[...uplift].filter(id=>current.has(id)).length,
          new_vs_current:[...uplift].filter(id=>!current.has(id)).length,
          current_plan_overlap:[...current].filter(id=>plan.has(id)).length,
          uplift_plan_overlap:[...uplift].filter(id=>plan.has(id)).length,
          list_opens:events.filter(e=>e.event==='list_opened').length,snapshot:'2026-11-01-demo'};
        for(const event of ['viewed','planned','contacted','deferred'])summary[event]=new Set(events.filter(e=>e.event===event&&uplift.has(e.hcp_id)).map(e=>e.hcp_id)).size;
        return summary;
      }
      throw Error('Unknown demo request.');
    },
    send:async function(event,hcp_id='',note='',list_type='uplift'){
      note=String(note).trim().slice(0,500);
      if(!['list_opened','viewed','planned','contacted','deferred'].includes(event))throw Error('Invalid action.');
      if(event!=='list_opened'&&!ids.has(hcp_id))throw Error('HCP not found.');
      if(event==='deferred'&&!note)throw Error('Please record a reason for deferring.');
      const events=read();events.push({id:events.length+1,at:new Date().toISOString(),snapshot:'2026-11-01-demo',hcp_id,event,note,list_type});write(events);return {ok:true};
    },
    export:function(){
      const columns=['id','at','snapshot','hcp_id','event','note','list_type'];
      const quote=v=>{let s=String(v??'');if(/^[=+@-]/.test(s))s="'"+s;return '"'+s.replace(/"/g,'""')+'"'};
      const csv=[columns.map(quote).join(','),...read().map(e=>columns.map(k=>quote(e[k])).join(','))].join('\r\n');
      const url=URL.createObjectURL(new Blob(['\ufeff',csv],{type:'text/csv;charset=utf-8'}));
      const a=document.createElement('a');a.href=url;a.download='fieldwise_browser_actions.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    }
  };
})();
