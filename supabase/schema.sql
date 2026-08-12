-- ============================================================
--  Esquema TablaRugby (Supabase / PostgreSQL)
--  Ejecutar en: Supabase Dashboard -> SQL Editor -> New query
-- ============================================================

-- ---------- Tablas ----------

create table if not exists public.torneos (
  id serial primary key,
  nombre text not null,               -- ej: "Torneo oficial 2010 2026"
  division text not null,             -- año de nacimiento: "2010"
  temporada integer not null,         -- 2026
  corte_top integer default 7,        -- cuántos clasifican (coloreado de tabla)
  activa boolean default false,
  created_at timestamptz default now()
);

create table if not exists public.etapas (
  id serial primary key,
  torneo_id integer references public.torneos(id) on delete cascade,
  nombre text not null,               -- "Regular", "Clasificacion", "Oro", "Plata"
  orden integer not null default 0,
  unique (torneo_id, nombre)
);

create table if not exists public.equipos (
  id serial primary key,
  torneo_id integer references public.torneos(id) on delete cascade,
  nombre text not null,
  unique (torneo_id, nombre)
);

create table if not exists public.partidos (
  id serial primary key,
  etapa_id integer references public.etapas(id) on delete cascade,
  local_equipo_id integer references public.equipos(id),
  visitante_equipo_id integer references public.equipos(id),
  nro integer,                        -- ID del partido en bd.uar (anti-duplicado)
  resultado_local integer,            -- null si no jugado
  resultado_visitante integer,
  puntos_local integer,               -- puntos de torneo otorgados ([N])
  puntos_visitante integer,
  fecha_hora timestamp,               -- SIN zona horaria (coincide con la planilla)
  referee text,
  estado text default 'Pendiente',    -- Pendiente / En Curso / Cerrado
  unique (nro)
);

create table if not exists public.tarjetas (
  id serial primary key,
  partido_id integer references public.partidos(id) on delete set null,
  equipo_id integer references public.equipos(id),
  fecha timestamp,
  incidencia text,                    -- amarilla / roja / azul
  instancia text,
  rival text,
  momento text,
  detalle text
);

-- ---------- Índices ----------
create index if not exists idx_partidos_etapa on public.partidos(etapa_id);
create index if not exists idx_partidos_nro  on public.partidos(nro);
create index if not exists idx_partidos_equipos on public.partidos(local_equipo_id, visitante_equipo_id);
create index if not exists idx_tarjetas_equipo on public.tarjetas(equipo_id);

-- ---------- RLS (Row Level Security) ----------
alter table public.torneos  enable row level security;
alter table public.etapas   enable row level security;
alter table public.equipos  enable row level security;
alter table public.partidos enable row level security;
alter table public.tarjetas enable row level security;

-- Público: cualquiera puede LEER todo (con la anon key)
create policy "lectura publica torneos"  on public.torneos  for select using (true);
create policy "lectura publica etapas"   on public.etapas   for select using (true);
create policy "lectura publica equipos"  on public.equipos  for select using (true);
create policy "lectura publica partidos" on public.partidos for select using (true);
create policy "lectura publica tarjetas" on public.tarjetas for select using (true);

-- Solo admins (rol 'admin' en app_metadata) pueden escribir.
-- Cualquier petición anónima o sin ese claim es rechazada.
create policy "escritura admin torneos"  on public.torneos  for all using (coalesce((auth.jwt() -> 'app_metadata' ->> 'role'), '') = 'admin');
create policy "escritura admin etapas"   on public.etapas   for all using (coalesce((auth.jwt() -> 'app_metadata' ->> 'role'), '') = 'admin');
create policy "escritura admin equipos"  on public.equipos  for all using (coalesce((auth.jwt() -> 'app_metadata' ->> 'role'), '') = 'admin');
create policy "escritura admin partidos" on public.partidos for all using (coalesce((auth.jwt() -> 'app_metadata' ->> 'role'), '') = 'admin');
create policy "escritura admin tarjetas" on public.tarjetas for all using (coalesce((auth.jwt() -> 'app_metadata' ->> 'role'), '') = 'admin');

-- ============================================================
--  (OPCIONAL) Después de crear tu admin en Authentication,
--  corré esto reemplazando tu email para darle rol de admin:
--
--  update auth.users
--  set raw_app_meta_data =
--      jsonb_set(coalesce(raw_app_meta_data, '{}'::jsonb), '{role}', '"admin"')
--  where email = 'TU_EMAIL@EJEMPLO.COM';
-- ============================================================