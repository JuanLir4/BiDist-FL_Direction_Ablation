# BiDist-FL Direction Ablation

## Execução no Google Colab

No Colab, clone o repositório, instale as dependências e execute o experimento
diretamente na sessão:

```python
!git clone https://github.com/JuanLir4/BiDist-FL_Direction_Ablation.git
%cd BiDist-FL_Direction_Ablation
!pip install -r requirements.txt
!python ablation_direction.py --datasets cifar10 --device auto --out results
```

Para salvar os dados ou resultados no Google Drive, use os caminhos montados
nos argumentos `--data-root` e `--out`. Para validar rapidamente a execução:

```python
!python ablation_direction.py --datasets cifar10 --quick --device auto
```

O estudo compara somente `c2s_only` e `bidirectional`. O CIFAR-10 usa a
arquitetura `ConvNIDD`; MNIST e Fashion-MNIST continuam usando `Net`.
