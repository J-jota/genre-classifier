const select = document.getElementById("filmes");
const filtro = document.getElementById("filtro");
const selectAlgoritmo = document.getElementById("algoritmo");
const detalhe = document.getElementById("detalhe");
const resultado = document.getElementById("resultado");
const erro = document.getElementById("erro");

let todosFilmes = [];

async function carregarFilmes() {
  const [respFilmes, respModelo] = await Promise.all([
    fetch("/api/movies"),
    fetch("/api/model"),
  ]);
  if (!respFilmes.ok) throw new Error("Falha ao carregar filmes.");
  todosFilmes = await respFilmes.json();
  preencherLista(todosFilmes);

  if (respModelo.ok) {
    const info = await respModelo.json();
    for (const nome of info.algoritmos_disponiveis) {
      const opt = document.createElement("option");
      opt.value = nome;
      opt.textContent = nome + (nome === info.modelo_vencedor ? " (vencedor)" : "");
      selectAlgoritmo.appendChild(opt);
    }
    selectAlgoritmo.value = info.modelo_vencedor;
    document.getElementById("info-algoritmo").textContent =
      "Melhor modelo por F1 macro no teste: " + info.modelo_vencedor;
  }
}

function preencherLista(filmes) {
  select.innerHTML = "";
  for (const f of filmes) {
    const opt = document.createElement("option");
    opt.value = f.id;
    opt.textContent = `${f.title} (${f.genero_real})`;
    select.appendChild(opt);
  }
}

filtro.addEventListener("input", () => {
  const termo = filtro.value.toLowerCase();
  preencherLista(todosFilmes.filter((f) => f.title.toLowerCase().includes(termo)));
  resultado.classList.add("oculto");
});

select.addEventListener("change", async () => {
  const id = Number(select.value);
  if (!id) return;
  erro.classList.add("oculto");
  const resp = await fetch(`/api/movies/${id}`);
  if (!resp.ok) {
    mostrarErro("Não foi possível carregar o filme.");
    return;
  }
  const filme = await resp.json();
  document.getElementById("titulo-filme").textContent = filme.title;
  document.getElementById("genero-real").textContent = filme.genero_real;
  document.getElementById("sinopse").textContent = filme.overview;
  detalhe.classList.remove("oculto");
  resultado.classList.add("oculto");
});

document.getElementById("btn-prever").addEventListener("click", async () => {
  const id = Number(select.value);
  if (!id) return;
  erro.classList.add("oculto");
  const btn = document.getElementById("btn-prever");
  btn.disabled = true;
  btn.textContent = "Prevendo...";
  try {
    const resp = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        movie_id: id,
        algorithm: selectAlgoritmo.value,
      }),
    });
    const dados = await resp.json();
    if (!resp.ok) throw new Error(dados.detail || "Erro na previsão.");

    document.getElementById("algoritmo-usado").textContent = dados.algoritmo;
    document.getElementById("genero-previsto").textContent = dados.genero_previsto;
    const cont = document.getElementById("probabilidades");
    cont.innerHTML = "";
    const ordenadas = Object.entries(dados.probabilidades).sort((a, b) => b[1] - a[1]);
    for (const [classe, prob] of ordenadas) {
      const linha = document.createElement("div");
      linha.className = "barra-linha";
      linha.innerHTML = `
        <span class="barra-rotulo">${classe}</span>
        <div class="barra-fundo">
          <div class="barra-preenchida" style="width:${(prob * 100).toFixed(1)}%"></div>
        </div>
        <span class="barra-valor">${(prob * 100).toFixed(1)}%</span>`;
      cont.appendChild(linha);
    }
    resultado.classList.remove("oculto");
  } catch (e) {
    mostrarErro(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Prever gênero pela sinopse";
  }
});

function mostrarErro(msg) {
  erro.textContent = msg;
  erro.classList.remove("oculto");
}

carregarFilmes().catch((e) => mostrarErro(e.message));
