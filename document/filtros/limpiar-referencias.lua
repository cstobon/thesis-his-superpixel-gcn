-- Pandoc no interpreta el LaTeX de algunos campos del .bib, como el tipo de
-- tesis que Better BibTeX exporta como {{Tesis de maestr\'ia}}. Este filtro
-- lee la bibliografía antes de citeproc, limpia esos campos (llaves y acentos
-- de LaTeX) y se la entrega a citeproc ya corregida.
local acentos = {
  ["\\'a"]="á", ["\\'e"]="é", ["\\'i"]="í", ["\\'o"]="ó", ["\\'u"]="ú",
  ["\\'A"]="Á", ["\\'E"]="É", ["\\'I"]="Í", ["\\'O"]="Ó", ["\\'U"]="Ú",
  ["\\~n"]="ñ", ["\\~N"]="Ñ", ['\\"u']="ü",
}
local function limpiar(texto)
  for k, v in pairs(acentos) do
    texto = texto:gsub(k:gsub("%p", "%%%0"), v)
  end
  return (texto:gsub("[{}]", ""))
end

function Pandoc(doc)
  if not doc.meta.bibliography then return nil end
  local refs = pandoc.utils.references(doc)
  for _, ref in ipairs(refs) do
    if ref.genre then
      ref.genre = limpiar(pandoc.utils.stringify(ref.genre))
    end
  end
  doc.meta.references = refs
  doc.meta.bibliography = nil
  return doc
end
