from torch import *
from data import *
from decoder import *
from encoder import *
from adain import *


def main():
    test_data()
    test_decoder()
    test_encoder()
    test_adain()


def test_data():
    transforms = get_transforms(True)

    coco_path = "datasets/train2017"
    coco_annot = "datasets/annotations/captions_train2017.json"
    wiki_path = "datasets/wikiart"
    dataset = StyleTransferDataset(coco_path, coco_annot, wiki_path, transform = transforms)
    print(f"Dataset length: {len(dataset)}, Wiki length: {len(dataset.wiki)}, COCO length: {len(dataset.coco)}")

    content, style = dataset[0]
    print(f"1st content img: {type(content)}, {content.shape}")
    print(f"1st style img: {type(style)}, {style.shape}")

    dataset = IterableStyleTransferDataset(coco_path, coco_annot, wiki_path, transform = transforms)
    print(f"Dataset length: {len(dataset)}, Wiki length: {len(dataset.wiki)}, COCO length: {len(dataset.coco)}")

    a = DataLoader(dataset, num_workers = 2)
    it = iter(a)
    content1, style1 = next(it)
    content2, style2 = next(it)
    assert not torch.all(content1 == content2)
    assert not torch.all(style1 == style2)
    print(f"1st content img: {type(content1)}, {content1.shape}")
    print(f"1st style img: {type(style1)}, {style1.shape}")

    """
    import pdb
    pdb.set_trace()
    """


    dataset = IterableStyleTransferDataset(coco_path, coco_annot, wiki_path,\
                                           transform = transforms, length = 111)
    j = 0
    for x, y in DataLoader(dataset, num_workers = 4): 
        j += len(x)
    assert j == 111, j


def test_adain():
    batch_size = 8
    channels = 5
    w = 6
    h = 6
    source = torch.randint(-5, 5, (batch_size, channels, w, h)).float()
    target = torch.randint(-5, 5, (batch_size, channels, w, h)).float()

    stylized_source = adain(source, target)

    target = target.view(batch_size, channels, -1)
    stylized_source = stylized_source.view(batch_size, channels, -1)

    # check variances the same
    target_variances = target.var(-1, unbiased = False)
    stylized_variances = stylized_source.var(-1, unbiased = False)
    assert torch.all(torch.abs(stylized_variances - target_variances) < 1e-4), \
            (target_variances, stylized_variances)
    print("Vars match")

    # check means the same
    target_means = target.mean(-1)
    stylized_means = stylized_source.mean(-1)
    assert torch.all(torch.abs(stylized_means - target_means) < 1e-4)
    print("Means match")

    # check values are just rescaled / shifted
    stylized_reshaped = stylized_source.view(batch_size, channels, w, h)
    diff = F.instance_norm(source) - F.instance_norm(stylized_reshaped)
    assert torch.all(torch.abs(diff) < 1e-4)
    print("Vals match")


def test_encoder():
    encoder = VGG19Encoder()
    print(encoder)
    encoder.freeze()
    sample_input = torch.ones((1, 3, 256, 256))
    sample_output = encoder(sample_input)
    print(f"Input shapes: {sample_input.shape}")
    print(f"Output shapes: {[i.shape for i in sample_output]}")



def test_decoder():
    encoder = VGG19Encoder()
    decoder = Decoder()


    sample_input = torch.ones((2, 3, 256, 256))
    outputs = encoder(sample_input)
    output = decoder(outputs[-1])

    print(f"Input shape: {sample_input.shape}")
    print(f"Intermediate shapes: {[output.shape for output in outputs]}")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()
