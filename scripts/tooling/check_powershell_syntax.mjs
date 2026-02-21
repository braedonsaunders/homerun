import fs from 'node:fs'
import path from 'node:path'

import Parser from 'tree-sitter'
import powershell from 'tree-sitter-powershell'

function collectErrors(node, errors) {
  const missing = typeof node.isMissing === 'function' ? node.isMissing() : Boolean(node.isMissing)
  if (node.type === 'ERROR' || missing) {
    errors.push(node)
  }
  for (let i = 0; i < node.childCount; i += 1) {
    collectErrors(node.child(i), errors)
  }
}

function parseFile(filePath) {
  const parser = new Parser()
  parser.setLanguage(powershell)

  const source = fs.readFileSync(filePath, 'utf8')
  const tree = parser.parse(source)
  const root = tree.rootNode
  const errors = []
  collectErrors(root, errors)
  const hasError = typeof root.hasError === 'function' ? root.hasError() : Boolean(root.hasError)

  if (!hasError && errors.length === 0) {
    return []
  }

  return errors.map((node) => {
    const row = Number(node.startPosition?.row ?? 0) + 1
    const column = Number(node.startPosition?.column ?? 0) + 1
    return `${filePath}:${row}:${column} ${node.type}`
  })
}

const files = process.argv.slice(2)
if (files.length === 0) {
  console.error('Usage: node check_powershell_syntax.mjs <file1.ps1> [file2.ps1 ...]')
  process.exit(2)
}

let failures = 0
for (const raw of files) {
  const filePath = path.resolve(raw)
  if (!fs.existsSync(filePath)) {
    console.error(`${filePath}:1:1 MISSING_FILE`)
    failures += 1
    continue
  }

  const errors = parseFile(filePath)
  if (errors.length > 0) {
    for (const line of errors) {
      console.error(line)
    }
    failures += 1
  }
}

if (failures > 0) {
  process.exit(1)
}

for (const raw of files) {
  console.log(`OK ${path.resolve(raw)}`)
}
